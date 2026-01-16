# -------------------------------Imports-----------------------------------------
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import scipy.stats as stats
import pandas as pd
import optuna
import json
from torch.utils.data import IterableDataset, DataLoader
import gc
import copy

# -------------------------Script Settings---------------------------------------
# OPTUNA TUNING
use_saved_params_1 = True  # Set to True to load saved parameters instead of running Optuna
use_saved_params_14 = True  # Set to True to load saved parameters instead of running Optuna

# LOAD MODEL
load_model_1 = True  # Set to True to load directly the retrained model for dim(X)=1 instead of running the retraining
load_model_14 = True  # Set to True to load directly the retrained model for dim(X)=14 instead of running the retraining

# GROUND TRUTH COMPUTATION
compute_ground_truth = False  # Set to True to load saved ground truth value instead of computing it

torch.set_default_dtype(torch.float64)

BASE_DIR = os.getcwd()
GT_FILE = os.path.join(BASE_DIR, "PEMC_ground_truth.json")
PARAMS_FILE_1 = os.path.join(BASE_DIR, "PEMC_1_best_params.json")
PARAMS_FILE_14 = os.path.join(BASE_DIR, "PEMC_14_best_params.json")
MODEL_FILE_1 = os.path.join(BASE_DIR, "PEMC_1_trained_model.pth")
MODEL_FILE_14 = os.path.join(BASE_DIR, "PEMC_14_trained_model.pth")

# ---------------------------Seed Settings---------------------------------------
def set_all_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ All seeds set to {seed}")
    if torch.cuda.is_available():
        print(f"✓ Running on: {torch.cuda.get_device_name(0)}")


# Set seeds
set_all_seeds(42)

# --------------------Save and load the best hyperparameters---------------------
def save_best_params(best_params, dim_X, filename):
    """
    Saves the best hyperparameters in a .json file.
    """
    params = {
        f"dim_X_{dim_X}": best_params
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Best hyperparameters for dim_X={dim_X} saved to {filename}")

def load_best_params(dim_X, filename):
    """
    Loads the best hyperparameters from a .json file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Hyperparameter file {filename} not found")
    with open(filename, 'r') as f:
        params = json.load(f)
    return params[f"dim_X_{dim_X}"]

# -----------------------------Simulations---------------------------------------
def simulate_X(N, W_dt, dim_X, device):
    """
    Simulates the stochastic component X.

    Arguments:
        N: number of paths to be simulated.
        W_dt: 2D tensor of Brownian increments.
        dim_X: dimension of X.
        device: device where the data is stored (CPU/GPU).
    """
    # Compute the number of consecutive Brownian increments to sum to obtain a component of X
    increments_partition_size = W_dt.shape[1] // dim_X

    # Compute X as the sum of "increments_partition_size" consecutive increments
    X = torch.zeros((N, dim_X), device=device)
    for i in range(dim_X):
        X[:, i] = torch.sum(W_dt[:, i * increments_partition_size:(i + 1) * increments_partition_size], dim=1)

    return X

def simulate_arithmetic_asian_option_payoff(N, sampling_freq, dt, W_dt, theta, device):
    """
    Simulates the payoff of an arithmetic Asian call option.

    Arguments:
        N: number of paths to be simulated.
        sampling_freq: sampling frequency.
        dt: time discretization step.
        W_dt: 2D tensor of Brownian increments.
        theta: vector of parameters.
        device: device where the data is stored (CPU/GPU).
    """
    # Simulate the tensor of log-returns according to GBM
    log_returns = torch.zeros((N, sampling_freq + 1), device=device)

    log_returns[:, 1:] = torch.cumsum(
        (theta[:, 0:1] - 0.5 * theta[:, 2:3] ** 2) * dt + theta[:, 2:3] * W_dt,
        dim=1)

    # Compute the spot price according to the simulation
    S = theta[:, 1:2] * torch.exp(log_returns)

    # Compute the payoff of the arithmetic Asian call option
    mean_S = torch.mean(S[:, 1:], dim=1, keepdim=True)
    K = theta[:, 3:4]
    payoff = torch.max(torch.zeros_like(mean_S, device=device), mean_S - K)

    return payoff

def simulate_geometric_asian_option_payoff(N, sampling_freq, dt, W_dt, theta, device):
    '''
    Simulates the payoff of a geometric Asian call option.

    Arguments:
        N: number of paths to be simulated.
        sampling_freq: sampling frequency.
        dt: time discretization step.
        W_dt: 2D tensor of Brownian increments.
        theta: vector of parameters.
        device: device where the data is stored (CPU/GPU).    
    '''
    # Simulate the tensor of log-returns according to GBM
    log_returns = torch.zeros((N, sampling_freq + 1), device=device)
    log_returns[:, 1:] = torch.cumsum(
        (theta[:, 0:1] - 0.5 * theta[:, 2:3] ** 2) * dt + theta[:, 2:3] * W_dt,
        dim=1)

    # Compute the payoff of the geometric Asian call option
    geom_mean_S = theta[:, 1:2] * torch.exp(torch.mean(log_returns[:, 1:], dim=1, keepdim=True))
    K = theta[:, 3:4]
    payoff = torch.max(torch.zeros_like(geom_mean_S, device=device), geom_mean_S - K)

    return payoff

def geometric_asian_option_closed_form_expected_payoff(r, S0, sigma, K, T, n):
    """
    Computes the closed-form price for a geometric Asian call option.

    Arguments:
        r: risk-free rate.
        S0: initial spot price of the underlying asset.
        sigma: volatilty of the underlying asset.
        K: strike price of the geometric Asian option.
        T: maturity of the option.
        n: sampling frequency.
    """
    sigma_n = sigma * np.sqrt((2 * n + 1) * (n + 1) / (6 * n ** 2))
    mu_n = (r - sigma ** 2 / 2) * (n + 1)/ (2 * n) + 0.5 * sigma_n **2
    d1 = (np.log(S0 / K) + (mu_n + sigma_n ** 2 / 2) * T) / (sigma_n * np.sqrt(T))
    d2 = d1 - sigma_n * np.sqrt(T)
    expected_payoff = (S0 * np.exp(mu_n * T) * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))

    return expected_payoff

def scale_theta(theta_tensor, intervals):
    """
    Scales theta with min-max scaling.

    Arguments:
        theta_tensor: tensor to scale.
        intervals: sampling intervals of the parameters of the vector to scale.
    """
    theta_scaled = torch.zeros_like(theta_tensor)
    for i, (low, high) in enumerate(intervals):
        theta_scaled[:, i] = (theta_tensor[:, i] - low) / (high - low)
    return theta_scaled


def scale_X(X, dt, sampling_freq, dim_X):
    """
    Scales X with standard scaling.

    Arguments:
        X: tensor to scale.
        dt: time discretization step.
        sampling_freq: sampling frequency.
        dim_X: dimension of X.
    """
    increments_partition_size = sampling_freq // dim_X
    variance = increments_partition_size * dt
    std_X = np.sqrt(variance)

    return X / std_X

# -----------------------------------Dataset Generation-------------------------------------
class PEMCDataset(IterableDataset):
    """
    Creates the training dataset.
    """
    def __init__(self, num_samples, sampling_freq, intervals, dt, dim_X, device, batch_size):
        """
        Arguments:
            num_samples: total number of training samples.
            sampling_freq: sampling frequency.
            intervals: intervals used for uniform sampling of theta.
            dt: time discretization step.
            dim_X: dimension of X.
            device: device where the data is stored (CPU/GPU).
            batch_size: size of the training batch.
        """
        super(PEMCDataset, self).__init__()
        self.device = device
        self.num_samples = num_samples
        self.sampling_freq = sampling_freq
        self.intervals = intervals
        self.dt = dt
        self.dim_X = dim_X
        self.n_params = len(intervals)
        self.batch_size = batch_size
        self.batches_per_epoch = self.num_samples // self.batch_size + (self.num_samples % self.batch_size > 0)

    def __iter__(self):
        for batch_idx in range(self.batches_per_epoch):

            # Computation of the batch size in order to manage the last batch, that could be smaller than the previous ones
            current_batch_size = int(min(self.batch_size, self.num_samples - batch_idx * self.batch_size))

            # Generate batch on-the-fly
            theta = torch.zeros((current_batch_size, self.n_params), device=self.device)
            for i, (low, high) in enumerate(self.intervals):
                theta[:, i].uniform_(low, high)

            W_dt = torch.normal(0.0, float(np.sqrt(self.dt)), size=(current_batch_size, self.sampling_freq), device=self.device)

            payoff = simulate_arithmetic_asian_option_payoff(current_batch_size, self.sampling_freq, self.dt, W_dt, theta, self.device)
            X = simulate_X(current_batch_size, W_dt, self.dim_X, self.device)

            # Scale theta and X
            theta_scaled = scale_theta(theta, self.intervals)
            X_scaled = scale_X(X, self.dt, self.sampling_freq, self.dim_X)

            yield theta_scaled, X_scaled, payoff

class ValidationDataset(IterableDataset):
    """
    Creates the validation dataset.
    """
    def __init__(self, num_samples, sampling_freq, intervals, dt, dim_X, device):
        """
        Arguments:
            num_samples: total number of training samples.
            sampling_freq: sampling frequency.
            intervals: intervals used for uniform sampling of theta.
            dt: time discretization step.
            dim_X: dimension of X.
            device: device where the data is stored (CPU/GPU).
        """
        super(ValidationDataset, self).__init__()
        self.device = device
        self.num_samples = num_samples
        self.sampling_freq = sampling_freq
        self.intervals = intervals
        self.dt = dt
        self.dim_X = dim_X
        self.n_params = len(intervals)

        # Generate all theta values
        self.theta = torch.zeros((self.num_samples, self.n_params), device=self.device)
        for i, (low, high) in enumerate(self.intervals):
            self.theta[:, i].uniform_(low, high)

        self.W_dt = torch.normal(0.0, float(np.sqrt(self.dt)),
                                  size=(self.num_samples, self.sampling_freq),
                                  device=self.device)

        # Generate all payoffs and X
        self.payoff = simulate_arithmetic_asian_option_payoff(
            self.num_samples, self.sampling_freq, self.dt,
            self.W_dt, self.theta, self.device
        )
        self.X = simulate_X(self.num_samples, self.W_dt, self.dim_X, self.device)

        # Scale theta and X
        self.theta_scaled = scale_theta(self.theta, self.intervals)
        self.X_scaled = scale_X(self.X, self.dt, self.sampling_freq, self.dim_X)

    def __iter__(self):
        yield self.theta_scaled, self.X_scaled, self.payoff
        
# --------------------------------Model------------------------------------------
class PEMCNetwork(nn.Module):
    """
    Initializes the model.
    """
    def __init__(self, x_dim, theta_hidden=256, combined_hidden=256, output_dim=1):
        """
        Arguments:
            x_dim: dimension of X.
            theta_hidden: number of neurons in each hidden layer of the theta network branch.
            combined_hidden: number of neurons in each hidden layer of the combined network.
            output_dim: dimension of the network's output.
        """
        super(PEMCNetwork, self).__init__()

        self.x_dim = x_dim

        # Theta network branch
        self.theta_branch = nn.Sequential(
            nn.Linear(4, theta_hidden),
            nn.BatchNorm1d(theta_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(theta_hidden, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # X network branch
        x_hidden = max(32, 2 * x_dim)
        self.x_branch = nn.Sequential(
            nn.Linear(x_dim, x_hidden),
            nn.Dropout(0.5),
            nn.Linear(x_hidden, x_hidden),
            nn.Dropout(0.5)
        )

        # Combined network
        combined_input_dim = 10 + x_hidden

        self.combined_fc1 = nn.Linear(combined_input_dim, combined_hidden)
        self.combined_bn1 = nn.BatchNorm1d(combined_hidden)

        self.combined_fc2 = nn.Linear(combined_hidden, combined_hidden)
        self.combined_bn2 = nn.BatchNorm1d(combined_hidden)

        # Skip connection dimension management
        if combined_input_dim != combined_hidden:
          self.skip_connection = nn.Sequential(
              nn.Linear(combined_input_dim, combined_hidden),
              nn.BatchNorm1d(combined_hidden)
          )
        else:
          self.skip_connection = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Output layer
        self.output_layer = nn.Linear(combined_hidden, output_dim)

        self.apply(self._init_weights)

    def forward(self, theta, x):
        # Process through branches
        theta_out = self.theta_branch(theta)
        x_out = self.x_branch(x)

        # Concatenate features
        combined = torch.cat([theta_out, x_out], dim=1)

        residual = self.skip_connection(combined)

        # First combined layer
        out = self.combined_fc1(combined)
        out = self.combined_bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Second combined layer
        out = self.combined_fc2(out)
        out = self.combined_bn2(out)

        # Skip connection and final ReLU
        out += residual
        out = F.relu(out)
        out = self.dropout(out)

        # Final prediction
        return self.output_layer(out)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Initialize weights
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

            # Initialize all bias values to zero
            if m.bias is not None:
                nn.init.zeros_(m.bias)
# -------------------------Training------------------------------------------------
class training:
    """
    Trains the model.
    """
    def __init__(self, model, Ntrain, batch_size, sampling_freq, intervals, dt, dim_X, lr=1e-3):
        """
        Arguments:
            model: "PEMCNetwork" object that represents the model used for training.
            Ntrain: total number of training samples.
            batch_size: size of the training batch.
            sampling_freq: sampling frequency.
            intervals: intervals used for uniform sampling of theta.
            dt: temporal discretization step.
            dim_X: dimension of X.
            lr: learning rate.
        """        
        # Use GPU, if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Store all the useful parameters
        self.Ntrain = Ntrain
        self.batch_size = batch_size
        self.sampling_freq = sampling_freq
        self.intervals = intervals
        self.dt = dt
        self.dim_X = dim_X

        # Model and training setup
        self.model = model.to(self.device).double()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Early-stopping variables
        self.best_mare = float('inf')
        self.loss_at_best_mare = float('inf')
        self.best_model_state = None

        # Initialize the training dataset and the DataLoader
        self.train_dataset = PEMCDataset(Ntrain, sampling_freq, intervals, dt, dim_X, self.device, batch_size=self.batch_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=None)

    def validate(self, val_loader):
        """
        Compute MSE and modified MARE on the validation dataset.
        
        Arguments:
            val_loader: DataLoader for the validation set.
        """
        self.model.eval()

        # Compute the validation losses on the whole validation set
        with torch.no_grad():
            theta_val, x_val, y_val = next(iter(val_loader))

            # Compute the MSE loss to be used for hyperparameter tuning
            output = self.model(theta_val, x_val)
            loss = self.criterion(output, y_val)

            # Compute the modified MARE loss to be used for early-stopping
            total_samples = theta_val.size(0)
            prediction = output.sum().item()
            target = y_val.sum().item()
            avg_pred = prediction / total_samples
            avg_target = target / total_samples
            denom = abs(avg_target) if abs(avg_target) > 1e-9 else 1e-9
            mare_diagnostic = abs(avg_pred - avg_target) / denom

            return loss.item(), mare_diagnostic

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
    def __init__(self, dt, sampling_freq, dim_X, intervals):
        """
        Arguments:
            dt: time discretization step.
            sampling_freq: sampling frequency.
            dim_X: dimension of X.
            intervals: intervals used for uniform sampling of theta.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dt = dt
        self.sampling_freq = sampling_freq
        self.dim_X = dim_X
        self.intervals = intervals

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
                W_dt = torch.normal(0.0, float(np.sqrt(self.dt)), size=(current_size, self.sampling_freq),
                                    device=self.device)
                payoff = simulate_arithmetic_asian_option_payoff(current_size, self.sampling_freq, self.dt, W_dt,
                                                                 theta_tensor[:int(current_size)], self.device)
                sum_payoffs += torch.sum(payoff)
        return (sum_payoffs / n).item()

    def evaluate_CV(self, n, theta):
        """
        Computes the CV estimator.

        Arguments:
            n: sample size.
            theta: vector of the evaluation parameters.
        """
        W_dt = torch.normal(0.0, float(np.sqrt(self.dt)), size=(int(n), self.sampling_freq), device=self.device)
        theta_tensor = torch.tensor(theta, device=self.device).repeat(int(n), 1)
        payoff_aritm = simulate_arithmetic_asian_option_payoff(int(n), self.sampling_freq, self.dt, W_dt, theta_tensor,
                                                               self.device)
        payoff_geom = simulate_geometric_asian_option_payoff(int(n), self.sampling_freq, self.dt, W_dt, theta_tensor,
                                                             self.device)
        expected_payoff_exact = geometric_asian_option_closed_form_expected_payoff(theta[0], theta[1], theta[2],
                                                                                   theta[3],
                                                                                   self.dt * self.sampling_freq,
                                                                                   self.sampling_freq)
        cv = torch.mean(payoff_aritm - payoff_geom).item() + expected_payoff_exact

        return cv

    def evaluate_PEMC(self, model, N, n, theta):
        """
        Computes the PEMC estimator.

        Arguments:
            model: "PEMCNetwork" object that represents the model used to compute the PEMC estimator.
            N: N=10n.
            n: sample size.
            theta: vector of the evaluation parameters.
        """  
        # Generate n paired samples (label, features)
        theta_tensor = torch.tensor(theta, device=self.device).repeat(int(n), 1)
        W_dt = torch.normal(0.0, float(np.sqrt(self.dt)), size=(int(n), self.sampling_freq), device=self.device)
        f = simulate_arithmetic_asian_option_payoff(int(n), self.sampling_freq, self.dt, W_dt, theta_tensor,
                                                    self.device)
        X = simulate_X(int(n), W_dt, self.dim_X, self.device)

        # Scale theta and X
        theta_scaled = scale_theta(theta_tensor, self.intervals)
        X_scaled = scale_X(X, self.dt, self.sampling_freq, self.dim_X)

        # Generate N i.i.d. samples of X
        theta_tensor_tilda = torch.tensor(theta, device=self.device).repeat(int(N), 1)
        W_dt_tilda = torch.normal(0.0, float(np.sqrt(self.dt)), size=(int(N), self.sampling_freq), device=self.device)
        X_tilda = simulate_X(int(N), W_dt_tilda, self.dim_X, self.device)

        # Scale theta_tilda and X_tilda
        theta_tilda_scaled = scale_theta(theta_tensor_tilda, self.intervals)
        X_tilda_scaled = scale_X(X_tilda, self.dt, self.sampling_freq, self.dim_X)

        # Set the model to evaluation mode
        model.eval()

        # Run inference
        with torch.no_grad():
            g = model(theta_scaled, X_scaled)
            g_tilda = model(theta_tilda_scaled, X_tilda_scaled)

        # Compute PEMC estimator
        PEMC = torch.mean(f - g) + torch.mean(g_tilda)

        return PEMC.item()

# ----------------------------Optuna Optimization--------------------------------
# Sampling parameters
Ntrain = 128 * 10**4
sampling_freq = 252
intervals = [(0.01, 0.03), (80, 120), (0.05, 0.25), (90, 110)] #(r,S0,sigma,K)
dt = 1 / sampling_freq

# Optuna parameters
grid_search_epochs_1 = 150
grid_search_epochs_14 = 200
grid_search_patience_1 = 15
grid_search_patience_14 = 20
n_trials = 75

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the number of samples of the validation set
val_dim = int(Ntrain * 0.1)

def run_optuna_study(dim_X):

    # Initialize the validation set for the hyperparameter tuning
    hyperparameters_val_set = ValidationDataset(val_dim, sampling_freq, intervals, dt, dim_X, device)
    hyperparameters_loader = DataLoader(hyperparameters_val_set, batch_size=None)

    def objective(trial):
        model = None
        trainer = None
        try:
            batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048, 4096])
            theta_hidden = trial.suggest_int('theta_hidden', 16, 256)
            combined_hidden = trial.suggest_int('combined_hidden', 16, 256)

            epochs = grid_search_epochs_14 if dim_X == 14 else grid_search_epochs_1
            patience = grid_search_patience_14 if dim_X == 14 else grid_search_patience_1
            early_stopping_loader = early_stopping_loader_14 if dim_X == 14 else early_stopping_loader_1

            # Create the model
            model = PEMCNetwork(x_dim=dim_X, theta_hidden=theta_hidden, combined_hidden=combined_hidden)
            trainer = training(model, Ntrain, batch_size, sampling_freq, intervals, dt, dim_X, lr=1e-3)
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

# Skip training and load directly the best model for dim(X)=1
if load_model_1:
    print(f"Skipping training and loading model from {MODEL_FILE_1}...")

    if not os.path.exists(MODEL_FILE_1):
        raise FileNotFoundError(f"Model file {MODEL_FILE_1} not found, run training first!")

    # Load the best hyperparameters for dim(X)=1
    best_params_1 = load_best_params(1, PARAMS_FILE_1)

    # Create the model architecture
    model_1 = PEMCNetwork(x_dim=1, theta_hidden=best_params_1['theta_hidden'],
                          combined_hidden=best_params_1['combined_hidden'])

    model_1 = model_1.to(device).double()

    # Upload weights and biases for dim(X)=1
    state_dict = torch.load(MODEL_FILE_1, map_location=device)
    model_1.load_state_dict(state_dict)

    print("Model loaded successfully")

# Train the model for dim(X)=1
else:
    # Initialize the validation set for early-stopping for dim(X)=1
    early_stopping_val_set_1 = ValidationDataset(val_dim, sampling_freq, intervals, dt, 1, device)
    early_stopping_loader_1 = DataLoader(early_stopping_val_set_1, batch_size=None)

    # Load the best hyperparameters for dim(X)=1 and just do the final retraining
    if use_saved_params_1:
        print(f"Loading hyperparameters from input...")
        best_params_1 = load_best_params(1, PARAMS_FILE_1)

    # Run Optuna hyperparameter tuning for dim(X)=1
    else:
        print("Starting Optuna study for dim_X=1...")
        best_params_1 = run_optuna_study(1)
        save_best_params(best_params_1, 1, PARAMS_FILE_1)

    # Retrain with best hyperparameters for dim(X)=1
    print("Retraining with best hyperparameters for dim_X=1...")
    model_1 = PEMCNetwork(x_dim=1, theta_hidden=best_params_1['theta_hidden'],
                          combined_hidden=best_params_1['combined_hidden'])
    trainer_1 = training(model_1, Ntrain, best_params_1['batch_size'], sampling_freq, intervals, dt, 1, lr=1e-3)
    trainer_1.fit(num_epochs=grid_search_epochs_1, patience=grid_search_patience_1, val_loader=early_stopping_loader_1)

    print(f"Saving trained model to {MODEL_FILE_1}...")
    torch.save(model_1.state_dict(), MODEL_FILE_1)
    print("Model saved successfully")

    # Clean memory
    del early_stopping_val_set_1, early_stopping_loader_1
    del trainer_1.train_dataset
    del trainer_1.train_loader
    del trainer_1.optimizer

# Skip training and load directly the best model for dim(X)=14
if load_model_14:
    print(f"Skipping training and loading model from {MODEL_FILE_14}...")

    if not os.path.exists(MODEL_FILE_14):
        raise FileNotFoundError(f"Model file {MODEL_FILE_14} not found, run training first!")

    # Load the best hyperparameters for dim(X)=14
    best_params_14 = load_best_params(14, PARAMS_FILE_14)

    # Create the model architecture
    model_14 = PEMCNetwork(x_dim=14, theta_hidden=best_params_14['theta_hidden'],
                          combined_hidden=best_params_14['combined_hidden'])

    model_14 = model_14.to(device).double()

    # Upload weights and biases for dim(X)=14
    state_dict = torch.load(MODEL_FILE_14, map_location=device)
    model_14.load_state_dict(state_dict)

    print("Model loaded successfully")

# Train the model for dim(X)=14
else:
    # Initialize the validation set for early-stopping for dim(X)=14
    early_stopping_val_set_14 = ValidationDataset(val_dim, sampling_freq, intervals, dt, 14, device)
    early_stopping_loader_14 = DataLoader(early_stopping_val_set_14, batch_size=None)

    # Load the best hyperparameters for dim(X)=14 and just do the final retraining
    if use_saved_params_14:
        print(f"Loading hyperparameters from input...")
        best_params_14 = load_best_params(14, PARAMS_FILE_14)

    # Run Optuna hyperparameter tuning for dim(X)=14
    else:
        print("Starting Optuna study for dim_X=14...")
        best_params_14 = run_optuna_study(14)
        save_best_params(best_params_14, 14, PARAMS_FILE_14)

    # Retrain with best hyperparameters for dim(X)=14
    print("Retraining with best hyperparameters for dim_X=14...")
    model_14 = PEMCNetwork(x_dim=14, theta_hidden=best_params_14['theta_hidden'],
                          combined_hidden=best_params_14['combined_hidden'])
    trainer_14 = training(model_14, Ntrain, best_params_14['batch_size'], sampling_freq, intervals, dt, 14, lr=1e-3)
    trainer_14.fit(num_epochs=grid_search_epochs_14, patience=grid_search_patience_14, val_loader=early_stopping_loader_14)

    print(f"Saving trained model to {MODEL_FILE_14}...")
    torch.save(model_14.state_dict(), MODEL_FILE_14)
    print("Model saved successfully")

    # Clean memory
    del early_stopping_val_set_14, early_stopping_loader_14
    del trainer_14.train_dataset
    del trainer_14.train_loader
    del trainer_14.optimizer

# --------------------------Metrics Evaluation-----------------------------------
# Delete datasets to free memory
if 'hyperparameters_val_set' in globals(): del hyperparameters_val_set
if 'hyperparameters_loader' in globals(): del hyperparameters_loader
gc.collect()
torch.cuda.empty_cache()

# Evaluation parameters
num_runs = 300
n_values = [1000, 4000, 9000]
theta_eval = [0.02, 100, 0.2, 100] #(r,S0,sigma,K)
batch_eval = 2048*1000

# Set the seed for evaluation
set_all_seeds(42)

evaluator_1 = evaluation(dt, sampling_freq, 1, intervals)
evaluator_14 = evaluation(dt, sampling_freq, 14, intervals)
theta_tensor = torch.tensor(theta_eval, device=device).repeat(batch_eval, 1)

# Compute ground truth
if compute_ground_truth:
    print("Computing ground truth...")
    ground_truth = evaluator_1.evaluate_MC(int(2e9), theta_tensor, batch_eval)
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
rmseCV = np.zeros(len(n_values))
rmsePEMC_1 = np.zeros(len(n_values))
rmsePEMC_14 = np.zeros(len(n_values))

for i, n in enumerate(n_values):
    print(f"Evaluation with n={n}")
    errMC = 0
    errCV = 0
    errPEMC_1 = 0
    errPEMC_14 = 0

    for j in range(num_runs):
        MC = evaluator_1.evaluate_MC(n, theta_tensor, batch_eval)
        CV = evaluator_1.evaluate_CV(n, theta_eval)
        PEMC_1 = evaluator_1.evaluate_PEMC(model_1, 10 * n, n, theta_eval)
        PEMC_14 = evaluator_14.evaluate_PEMC(model_14, 10 * n, n, theta_eval)

        errMC += (MC - ground_truth) ** 2
        errCV += (CV - ground_truth) ** 2
        errPEMC_1 += (PEMC_1 - ground_truth) ** 2
        errPEMC_14 += (PEMC_14 - ground_truth) ** 2

    # Compute RMSE for current n
    rmseMC[i] = np.sqrt(errMC / num_runs)
    rmseCV[i] = np.sqrt(errCV / num_runs)
    rmsePEMC_1[i] = np.sqrt(errPEMC_1 / num_runs)
    rmsePEMC_14[i] = np.sqrt(errPEMC_14 / num_runs)

# Create a dataframe with the RMSE values for each estimator and value of n
errors = pd.DataFrame(
    data=[rmseMC, rmsePEMC_1, rmsePEMC_14, rmseCV],
    columns=[f'n={n}' for n in n_values],
    index=['Monte Carlo (MC)', 'PEMC (dim(X) = 1)', 'PEMC (dim(X) = 14)', 'Geometric CV']
)
print(errors)

# Compute the percentage reduction of PEMC with respect to MC
PEMC_1_reduction = np.zeros(len(n_values))
PEMC_14_reduction = np.zeros(len(n_values))
for i, n in enumerate(n_values):
  PEMC_1_reduction[i] = (errors[f'n={n}']['Monte Carlo (MC)'] - errors[f'n={n}']['PEMC (dim(X) = 1)']) / errors[f'n={n}']['Monte Carlo (MC)']
  PEMC_14_reduction[i] = (errors[f'n={n}']['Monte Carlo (MC)'] - errors[f'n={n}']['PEMC (dim(X) = 14)']) / errors[f'n={n}']['Monte Carlo (MC)']

# Create a datafame with the percentage reduction of PEMC with respect to MC
reductions = pd.DataFrame(
    data=[PEMC_1_reduction, PEMC_14_reduction],
    columns=[f'n={n}' for n in n_values],
    index=['PEMC (dim(X) = 1)', 'PEMC (dim(X) = 14)']
)
print(reductions.map(lambda x: f"{x:.3%}"))