# -------------------------------Imports-----------------------------------------
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import optuna
import json
from torch.utils.data import IterableDataset, DataLoader
from functools import partial
import gc
import copy

# ------------------------- Script Settings-------------------------------------
# OPTUNA TUNING
use_saved_params = True  # Set to True to load saved parameters instead of running Optuna

# LOAD MODEL
load_model = True  # Set to True to load directly the retrained model instead of running the retraining

# GROUND TRUTH COMPUTATION
compute_ground_truth = False  # Set to True to load saved ground truth value instead of computing it

torch.set_default_dtype(torch.float32)

BASE_DIR = os.getcwd()
GT_FILE = os.path.join(BASE_DIR, "heston_ground_truth.json")
PARAMS_FILE = os.path.join(BASE_DIR, "heston_best_params.json")
MODEL_FILE = os.path.join(BASE_DIR, "heston_trained_model.pth")

# ---------------------------Seed Settings---------------------------------------
def set_all_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ All seeds set to {seed}")
    if torch.cuda.is_available():
        print(f"✓ Running on: {torch.cuda.get_device_name(0)}")


# Set seeds
set_all_seeds(42)

# --------------------Save and load the best hyperparameters---------------------
def save_best_params(best_params, filename):
    """
    Saves the best hyperparameters in a .json file.
    """
    params = {
        f"params": best_params
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Best hyperparameters saved to {filename}")


def load_best_params(filename):
    """
    Loads the best hyperparameters from a .json file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Hyperparameter file {filename} not found")
    with open(filename, 'r') as f:
        params = json.load(f)
    return params[f"params"]

# -----------------------------Simulations---------------------------------------
def simulate_heston_payoff(r, kappa, eta, delta, rho, v0, T, dt, N):
    """
    Simulates the variance swap payoff and features X in the Heston model.
    
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_t^S
    dv_t = kappa * (eta - v_t) * dt + delta * sqrt(v_t) * dW_t^v

    Arguments:
        r, kappa, eta, delta: parameters of the Heston model.
        rho: correlation parameter of the Heston model.
        v0: intiial value of the variance process.
        T: time to maturity of the derivative.
        dt: time discretization step.
        N: number of paths to simulate.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sampling_freq = int(T / dt)
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

    # Convert parameters to tensors
    rho_tensor = torch.as_tensor(rho, device=device).unsqueeze(1)
    sqrt_1mrho2 = torch.sqrt(1 - rho_tensor ** 2)

    # Generate correlated Brownian increments
    Z = torch.randn(N, sampling_freq, 2, device=device)
    Z1 = Z[:, :, 0]
    Z2 = Z[:, :, 1]

    dWS = sqrt_dt * Z1
    dWv = sqrt_dt * (rho_tensor * Z1 + sqrt_1mrho2 * Z2)

    # Convert parameters to tensors
    v = torch.as_tensor(v0, device=device)
    r_tensor = torch.as_tensor(r, device=device)
    kappa_tensor = torch.as_tensor(kappa, device=device)
    eta_tensor = torch.as_tensor(eta, device=device)
    delta_tensor = torch.as_tensor(delta, device=device)

    sum_log_sq = torch.zeros(N, device=device)

    # Euler scheme with full truncation
    for i in range(sampling_freq):
        v_plus = torch.clamp(v, min=0.0)  
        sqrt_v = torch.sqrt(v_plus)

        log_ret = (r_tensor - 0.5 * v_plus) * dt + sqrt_v * dWS[:, i]
        dv = kappa_tensor * (eta_tensor - v_plus) * dt + delta_tensor * sqrt_v * dWv[:, i]

        v = v + dv 
        sum_log_sq += log_ret ** 2

    payoff = 100 * 252 * sum_log_sq / sampling_freq

    # Compute X
    WS_T = torch.sum(dWS, dim=1, keepdim=True)
    Wv_T = torch.sum(dWv, dim=1, keepdim=True)

    X = torch.cat([WS_T, Wv_T], dim=1)

    return payoff.unsqueeze(1), X

# -----------------------------------Dataset-------------------------------------
class PEMCDataset(IterableDataset):
    """
    Creates the training dataset.
    """
    def __init__(self, num_samples, intervals, dt, T, device, batch_size):
        """
        Arguments:
            num_samples: total number of training samples.
            intervals: intervals used for uniform sampling of theta.
            dt: time discretization step.
            T: time to maturity of the derivative.
            device: device where the data is stored (CPU/GPU).
            batch_size: size of the training batch.
        """
        super(PEMCDataset, self).__init__()
        self.device = device
        self.num_samples = num_samples
        self.intervals = intervals
        self.dt = dt
        self.T = T
        self.n_params = len(intervals)
        self.batch_size = batch_size
        self.batches_per_epoch = num_samples // batch_size + (num_samples % batch_size > 0)

    def __iter__(self):
        for batch_idx in range(self.batches_per_epoch):

            # Computation of the batch size in order to manage the last batch, that could be smaller than the previous ones
            current_batch_size = min(self.batch_size, self.num_samples - batch_idx * self.batch_size)

            # Generate batch on-the-fly
            theta = torch.zeros((current_batch_size, self.n_params), device=self.device)
            for i, (low, high) in enumerate(self.intervals):
                theta[:, i].uniform_(low, high)

            theta[:, 2] = theta[:, 2] ** 2
            theta[:, 4] = theta[:, 4] ** 2

            payoff, X = simulate_heston_payoff(theta[:, 1], theta[:, 3], theta[:, 4], theta[:, 5], theta[:, 6],
                                               theta[:, 2], self.T, self.dt, current_batch_size)

            yield theta, X, payoff


class ValidationDataset(IterableDataset):
    """
    Creates the validation dataset.
    """
    def __init__(self, num_samples, intervals, dt, T, device):
        """
        Arguments:
            num_samples: total number of training samples.
            intervals: intervals used for uniform sampling of theta.
            dt: time discretization step.
            T: time to maturity of the derivative.
            device: device where the data is stored (CPU/GPU).
        """
        super(ValidationDataset, self).__init__()

        self.device = device
        self.n_params = len(intervals)

        # Generate all theta values
        self.theta = torch.zeros((num_samples, self.n_params), device=self.device)
        for i, (low, high) in enumerate(intervals):
            self.theta[:, i].uniform_(low, high)

        self.theta[:, 2] = self.theta[:, 2] ** 2
        self.theta[:, 4] = self.theta[:, 4] ** 2

        # Generate all payoffs and X
        self.payoff, self.X = simulate_heston_payoff(self.theta[:, 1], self.theta[:, 3], self.theta[:, 4], self.theta[:, 5], self.theta[:, 6],
                                                     self.theta[:, 2], T, dt, num_samples)

    def __iter__(self):
        yield self.theta, self.X, self.payoff
# --------------------------------Model------------------------------------------
class PEMCNetwork(nn.Module):
    """
    Initializes the model.
    """
    def __init__(self, hidden_layers_1=1, hidden_layers_2=1, theta_dim=7, x_dim=2, hidden_dim_1=512, hidden_dim_2=256,
                 output_dim=1):
        """
        Arguments:
            hidden_layers_1: number of hidden layer of the first MLP block.
            hidden_layers_2: number of hidden layer of the second MLP block.
            theta_dim: number of elements of the vector of parameters theta.
            x_dim: number of elements of the vector X.
            hidden_dim_1: number of neurons in the hidden layers of the first MLP block.
            hidden_dim_2: number of neurons in the hidden layers of the second MLP block.
            output_dim: dimension of the network's output.
        """
        super(PEMCNetwork, self).__init__()

        input_dim = theta_dim + x_dim

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU()
        )

        # Skip connection dimension management
        self.skip_connection = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2)
        )

        # First MLP block
        layers_1 = []

        for _ in range(hidden_layers_1):
            layers_1.append(nn.Linear(hidden_dim_1, hidden_dim_1))
            layers_1.append(nn.BatchNorm1d(hidden_dim_1))
            layers_1.append(nn.ReLU())

        layers_1.append(nn.Linear(hidden_dim_1, hidden_dim_2))
        layers_1.append(nn.BatchNorm1d(hidden_dim_2))

        self.MLP_1 = nn.Sequential(*layers_1)

        # Second MLP block
        layers_2 = []

        for _ in range(hidden_layers_2):
            layers_2.append(nn.Linear(hidden_dim_2, hidden_dim_2))
            layers_2.append(nn.BatchNorm1d(hidden_dim_2))
            layers_2.append(nn.ReLU())

        layers_2.append(nn.Linear(hidden_dim_2, hidden_dim_2))
        layers_2.append(nn.BatchNorm1d(hidden_dim_2))

        self.MLP_2 = nn.Sequential(*layers_2)

        self.output_layer = nn.Linear(hidden_dim_2, output_dim)

        self.apply(self._init_weights)

    def forward(self, theta, x):

        input = torch.cat([theta, x], dim=1)

        out = self.input_layer(input)

        # Save residual for skip connections
        residual_1 = self.skip_connection(out)

        # First MLP block
        out = self.MLP_1(out)
        out += residual_1
        out = F.relu(out)

        residual_2 = out

        # Second MLP block
        out = self.MLP_2(out)
        out += residual_2
        out = F.relu(out)

        # Final prediction
        return self.output_layer(out)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Initialize weights
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

            # Initialize all bias values to zero
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# -------------------------Training--------------------------------------------------
class training:
    """
    Trains the model.
    """
    def __init__(self, model, Ntrain, batch_size, intervals, dt, T, lr=1e-3):
        """
        Arguments:
            model: "PEMCNetwork" object that represents the model used for training.
            Ntrain: total number of training samples.
            batch_size: size of the training batch.
            intervals: intervals used for uniform sampling of theta.
            dt: temporal discretization step.
            T: time to maturity of the derivative.
            lr: learning rate.
        """  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Store all the useful parameters
        self.Ntrain = Ntrain
        self.batch_size = batch_size
        self.intervals = intervals
        self.dt = dt
        self.T = T

        # Model and training setup
        self.model = model.to(self.device).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.criterion = nn.MSELoss()

        # Early-stopping variables
        self.best_mare = float('inf')
        self.loss_at_best_mare = float('inf')
        self.best_model_state = None

        # Initialize the training dataset and the DataLoader
        self.train_dataset = PEMCDataset(Ntrain, intervals, dt, T, self.device, batch_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=None)

    def validate(self, val_loader):
        """
        Compute MSE and modified MARE on the validation dataset.
        
        Arguments:
            val_loader: DataLoader for the validation set.
        """
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        sum_predictions, sum_targets = 0.0, 0.0

        # Compute the validation losses on the whole validation set
        with torch.no_grad():
            for theta_val, x_val, y_val in val_loader:

                output = self.model(theta_val, x_val)
                loss = self.criterion(output, y_val)

                batch_size = theta_val.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                sum_predictions += output.sum().item()
                sum_targets += y_val.sum().item()

            # Compute the MSE loss to be used for hyperparameter tuning
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

            # Compute the modified MARE loss to be used for early-stopping
            avg_pred = sum_predictions / total_samples
            avg_target = sum_targets / total_samples
            denom = abs(avg_target) if abs(avg_target) > 1e-9 else 1e-9
            mare_diagnostic = abs(avg_pred - avg_target) / denom

            return avg_loss, mare_diagnostic

    def fit(self, num_epochs, patience, val_loader, validation_freq=5, target_mare=0.01):
        """
        Trains the model on the training dataset.

        Arguments:
            num_epochs: number of training epochs.
            patience: patience for early-stopping.
            val_loader: DataLoader for the validation set.
            validation_freq: number of epochs that separate two different prints of the training losses and learning rate.
            target_mare: under this value of modified MARE the training procedure stops, since model training is optimal.
        """
        patience_counter = 0

        # Training loop
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            running_loss, total_train_samples = 0, 0

            for theta, x, y in self.train_loader:

                # Create a batch of the dataset and train the model on it
                self.optimizer.zero_grad()
                output = self.model(theta, x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                current_bs = theta.size(0)
                running_loss += loss.item() * current_bs
                total_train_samples += current_bs

            train_loss = running_loss / total_train_samples
            val_loss, val_mare = self.validate(val_loader)
            self.scheduler.step(val_mare) 

            is_improvement = False

            # Case of modified MARE <1%
            if val_mare < target_mare:
                tqdm.write(f"\n--> Target modified MARE reached: ({val_mare:.4%} < {target_mare:.1%})! Stop!")
                self.best_mare = val_mare
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                break

            # Case of improved modified MARE
            elif val_mare < self.best_mare:
                self.best_mare = val_mare
                self.loss_at_best_mare = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                is_improvement = True
                
            # Case of not improved modified MARE
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"Early stopping at epoch {epoch + 1}")
                    break

            # Validate at every epoch but print metrics every "validation_freq" epochs
            if (epoch + 1) % validation_freq == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                status = "(*)" if is_improvement else ""
                tqdm.write(
                    f"Ep {epoch + 1}: Train {train_loss:.6f} | Val {val_loss:.6f} | MARE {val_mare:.2%} | LR {current_lr:.2e} {status}"
                )

        # Load the best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Model loaded with best modified MARE state")

# -------------------------Evaluation----------------------------------------------
class evaluation:
    """
    Computes the MC, CV and PEMC estimators.
    """
    def __init__(self, T, dt):
        """
        Arguments:
            T: time to maturity of the derivative.
            dt: time discretization step.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.T = T
        self.dt = dt

    def evaluate_MC(self, n, theta_tensor, batch_size):
        """
        Computes the MC estimator.

        Arguments:
            n: sample size.
            theta_tensor: tensor that contains the evaluation parameters.
            batch_size: size of the batch used to compute the MC estimator.
        """   
        # Batched MC evaluation
        sum_payoffs = torch.tensor(0.0, device=self.device)
        num_batches = n // batch_size + (n % batch_size > 0)

        with torch.no_grad():
            for i in range(num_batches):

                # Accumulate the sum of payoffs for each batch
                current_size = int(min(batch_size, n - i * batch_size))
                batch_theta = theta_tensor[:current_size]
                payoff, _ = simulate_heston_payoff(batch_theta[:, 1], batch_theta[:, 3], batch_theta[:, 4],
                                                   batch_theta[:, 5], batch_theta[:, 6], batch_theta[:, 2], self.T,
                                                   self.dt, current_size)
                sum_payoffs += torch.sum(payoff)
        return (sum_payoffs / n).item()

    def evaluate_PEMC(self, model, N, n, theta_tensor, batch_size):
        """
        Computes the PEMC estimator.

        Arguments:
            model: "PEMCNetwork" object that represents the model used to compute the PEMC estimator.
            N: N=10n.
            n: sample size.
            theta_tensor: tensor that contains the evaluation parameters.
            batch_size: size of the batch used to compute the PEMC estimator.
        """ 
        batches_per_epoch_n = n // batch_size + (n % batch_size > 0)
        batches_per_epoch_N = N // batch_size + (N % batch_size > 0)
        sum_diff, sum_g_tilda = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        model.eval()

        with torch.no_grad():

            # Generate n paired samples (label, features)        
            for batch_idx in range(batches_per_epoch_n):
                current_batch_size_n = min(batch_size, n - batch_idx * batch_size)
                batch_theta_n = theta_tensor[:current_batch_size_n]
                f, x_n = simulate_heston_payoff(batch_theta_n[:, 1], batch_theta_n[:, 3], batch_theta_n[:, 4],
                                                batch_theta_n[:, 5], batch_theta_n[:, 6], batch_theta_n[:, 2], self.T,
                                                self.dt, current_batch_size_n)
                g = model(batch_theta_n, x_n)
                sum_diff += torch.sum(f - g)

            # Generate N i.i.d. samples of X
            for batch_idx in range(batches_per_epoch_N):
                current_batch_size_N = min(batch_size, N - batch_idx * batch_size)
                batch_theta_N = theta_tensor[:current_batch_size_N]
                _, x_N = simulate_heston_payoff(batch_theta_N[:, 1], batch_theta_N[:, 3], batch_theta_N[:, 4],
                                                batch_theta_N[:, 5], batch_theta_N[:, 6], batch_theta_N[:, 2], self.T,
                                                self.dt, current_batch_size_N)
                g_tilda = model(batch_theta_N, x_N)
                sum_g_tilda += torch.sum(g_tilda)
            
            # Compute PEMC estimator
            PEMC = sum_diff / n + sum_g_tilda / N
        return PEMC.item()

# ----------------------------Optuna Optimization--------------------------------
# Sampling parameters
Ntrain = 3 * 10 ** 6
intervals = [[50, 150], [0.01, 0.05], [0.1, 0.375], [1.5, 4.5], [0.1, 0.3], [0.1, 1.0],
             [-0.9, -0.2]]  # (S0,r,v0,k,eta,delta,rho)
T = 1
dt = 1 / 252

# Optuna parameters
epochs = 150
patience = 15
n_trials = 50

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the number of samples of the validation set
val_dim = int(Ntrain * 0.1)

def run_optuna_study():

    # Initialize the validation set for the hyperparameter tuning
    hyperparameters_val_set = ValidationDataset(val_dim, intervals, dt, T, device)
    hyperparameters_loader = DataLoader(hyperparameters_val_set, batch_size=None)

    def objective(trial):
        model = None
        trainer = None
        try:
            batch_size = trial.suggest_categorical('batch_size', [32768, 65536, 131072])
            hidden_layers_1 = trial.suggest_int('hidden_layers_1', 1, 2)
            hidden_layers_2 = trial.suggest_int('hidden_layers_2', 1, 2)
            lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

            # Create the model
            model = PEMCNetwork(hidden_layers_1=hidden_layers_1, hidden_layers_2=hidden_layers_2)
            trainer = training(model, Ntrain, batch_size, intervals, dt, T, lr=lr)
            trainer.fit(num_epochs=epochs, patience=patience, val_loader=early_stopping_loader)

            # Compute the MSE loss on the validation set for the hyperparameter tuning
            loss, _ = trainer.validate(hyperparameters_loader)

            return loss

        except Exception as e:
            print(f"Trial failed with error: {e}")
            raise e
        finally:
            # Clean memory
            if model is not None: del model
            if trainer is not None:
                if hasattr(trainer, 'train_dataset'): del trainer.train_dataset
                del trainer
            gc.collect()
            torch.cuda.empty_cache()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

# Skip training and load directly the best model
if load_model:
    print(f"Skipping training and loading model from {MODEL_FILE}...")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file {MODEL_FILE} not found, run training first!")

    # Load the best hyperparameters
    best_params = load_best_params(PARAMS_FILE)

    # Create the model architecture
    model = PEMCNetwork(hidden_layers_1=best_params['hidden_layers_1'], hidden_layers_2=best_params['hidden_layers_2'])
    model = model.to(device).float()

    # Upload weights and biases
    state_dict = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state_dict)

    print("Model loaded successfully")

# Train the model
else:
    # Initialize the validation set for early-stopping
    early_stopping_val_set = ValidationDataset(val_dim, intervals, dt, T, device)
    early_stopping_loader = DataLoader(early_stopping_val_set, batch_size=None)

    # Load the best hyperparameters and just do the final retraining
    if use_saved_params:
        print(f"Loading hyperparameters from input...")
        best_params = load_best_params(PARAMS_FILE)

    # Run Optuna hyperparameter tuning
    else:
        print("Starting Optuna study...")
        best_params = run_optuna_study()
        save_best_params(best_params, PARAMS_FILE)

    # Retrain with best hyperparameters
    print("Retraining with best hyperparameters...")
    model = PEMCNetwork(hidden_layers_1=best_params['hidden_layers_1'], hidden_layers_2=best_params['hidden_layers_2'])
    trainer = training(model, Ntrain, best_params['batch_size'], intervals, dt, T, lr=best_params['lr'])
    trainer.fit(num_epochs=epochs, patience=patience, val_loader=early_stopping_loader)

    print(f"Saving trained model to {MODEL_FILE}...")
    torch.save(model.state_dict(), MODEL_FILE)
    print("Model saved successfully")

    # Clean memory
    del early_stopping_val_set, early_stopping_loader
    del trainer.train_dataset
    del trainer.train_loader
    del trainer.optimizer

# --------------------------Metrics Evaluation-----------------------------------
# Delete datasets to free memory
if 'hyperparameters_val_set' in globals(): del hyperparameters_val_set
if 'hyperparameters_loader' in globals(): del hyperparameters_loader
gc.collect()
torch.cuda.empty_cache()

# Evaluation parameters
num_runs = 200
n_values = [1000, 2000, 4000, 6000, 8000, 10000, 20000]
theta_eval = [100, 0.02, 0.25 ** 2, 3.0, 0.2 ** 2, 0.6, -0.4]
batch_eval = 2048 * 1000

# Set the seed for evaluation
set_all_seeds(42)

evaluator = evaluation(T, dt)
theta_tensor = torch.tensor(theta_eval, device=device).repeat(batch_eval, 1)

# Compute ground truth
if compute_ground_truth:
    print("Computing ground truth...")
    ground_truth = evaluator.evaluate_MC(int(5e7), theta_tensor, batch_eval)
    data_to_save = {"ground_truth": ground_truth}
    os.makedirs(os.path.dirname(GT_FILE), exist_ok=True)
    with open(GT_FILE, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Ground truth saved to {GT_FILE}")

# Load ground truth from a .json file
else:
    if not os.path.exists(GT_FILE):
        raise FileNotFoundError(f"Ground truth file {GT_FILE} not found")
    with open(GT_FILE, 'r') as f:
        gt = json.load(f)
        ground_truth = gt['ground_truth']
    print(f"Ground truth loaded from input...")

# Print ground truth value
print(f"Ground_truth:{ground_truth}")

# Initialize arrays to store RMSE for each n
rmseMC = np.zeros(len(n_values))
rmsePEMC = np.zeros(len(n_values))

for i, n in enumerate(n_values):
    print(f"Evaluation with n={n}")

    # Reset error accumulators for each n
    errMC = 0
    errPEMC = 0

    for j in range(num_runs):
        MC = evaluator.evaluate_MC(n, theta_tensor, batch_eval)
        PEMC = evaluator.evaluate_PEMC(model, 10 * n, n, theta_tensor, batch_eval)

        errMC += (MC - ground_truth) ** 2
        errPEMC += (PEMC - ground_truth) ** 2

    # Compute RMSE for current n
    rmseMC[i] = np.sqrt(errMC / num_runs)
    rmsePEMC[i] = np.sqrt(errPEMC / num_runs)

# Create a dataframe with the RMSE values for each estimator and value of n
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
errors = pd.DataFrame(
    data=[rmseMC, rmsePEMC],
    columns=[f'n={n}' for n in n_values],
    index=['Monte Carlo (MC)', 'PEMC']
)
print(errors)

# Compute the percentage reduction of PEMC with respect to MC
PEMC_reduction = np.zeros(len(n_values))
for i, n in enumerate(n_values):
    PEMC_reduction[i] = (errors[f'n={n}']['Monte Carlo (MC)'] - errors[f'n={n}']['PEMC']) / errors[f'n={n}'][
        'Monte Carlo (MC)']

# Create a datafame with the percentage reduction of PEMC with respect to MC
reductions = pd.DataFrame(
    data=[PEMC_reduction],
    columns=[f'n={n}' for n in n_values],
    index=['PEMC']
)
print(reductions.map(lambda x: f"{x:.3%}"))
