# -------------------------------Imports-----------------------------------------
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import math
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
GT_FILE = os.path.join(BASE_DIR, "SLV_ground_truth.json")
PARAMS_FILE = os.path.join(BASE_DIR, "SLV_best_params.json")
MODEL_FILE = os.path.join(BASE_DIR, "SLV_trained_model.pth")

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
class SLVContext:
    """
    Pre-computes time grid, space grid, variables that do not depend on theta and the deterministic 'base' volatility surface.
    """
    def __init__(self, T, steps_x, steps_t, dt, device):
        """ 
        Arguments:
            T: time to maturity of the derivative.
            steps_x: number of points in the log-return grid.
            steps_t: number of points in the time grid.
            dt: temporal discretization step.
            device: device where the data is stored (CPU/GPU).
        """
        self.device = device
        self.dt = torch.tensor(dt, device=device)
        self.sqrt_dt = torch.sqrt(self.dt)
        self.sampling_freq = int(T / dt)

        self.steps_x = steps_x
        self.steps_t = steps_t
        self.sqrt_T = torch.tensor(T, device=device).sqrt()

        p = torch.tensor([0.3, 0.5, 0.2], device=device)
        tau = torch.tensor([0.4, 0.3, 0.6], device=device)

        # Create time grid
        self.tmin = 0.0
        self.tmax = T - 1e-3
        self.t_grid = torch.linspace(self.tmin, self.tmax, steps=steps_t, device=device)

        # Create log-price grid
        self.xmin = -6.0
        self.xmax = 6.0
        self.x_grid = torch.linspace(self.xmin, self.xmax, steps=steps_x, device=device)

        # Computation of sigma_base
        ttm = (T - self.t_grid).view(1, steps_t, 1)
        p_expanded = p.view(1, 1, 3)
        tau_expanded = tau.view(1, 1, 3)
        x_expanded = self.x_grid.view(steps_x, 1, 1)
        denom_arg = 2.0 * ttm * (tau_expanded ** 2)
        exponent = - x_expanded ** 2 / denom_arg - ttm * (tau_expanded ** 2) / 8.0

        # Log-sum-exp trick to avoid underflow
        max_exp = torch.max(exponent, dim=2, keepdim=True)[0]
        exp_rel = torch.exp(exponent - max_exp)

        numerator_rel = torch.sum(p_expanded * tau_expanded * exp_rel, dim=2)
        denominator_rel = torch.sum(exp_rel * p_expanded / tau_expanded, dim=2)
        self.sigma_base = torch.sqrt(numerator_rel / denominator_rel)

        # Compute time arrays for simulation loop
        time_steps = torch.arange(self.sampling_freq, device=device)
        self.t_arr = self.dt * time_steps

        # Compute the normalized time for grid_sample (it has to be in range [-1, 1])
        self.t_norm_arr = 2 * (self.t_arr - self.tmin) / (self.tmax - self.tmin) - 1


def simulate_slv(ctx, epsilon, r, v0, kappa, delta, rho, train=True, only_inputs=False):
    """
    Simulates X and/or the grid sigma and/or the payoff according to SLV model:
   
    dS_t = r * S_t * dt + sigma(S_t, t) * exp(v_t) * S_t * dW_t^S
    dv_t = kappa * (eta_t - v_t) * dt + delta * dW_t^v

    Arguments:
        ctx: SLVContext object.
        epsilon: standard deviation of the perturbated volatility surface.
        v0: initial value of the variance process.
        r, kappa, delta: parameters of the SLV model. 
        rho: correlation parameter in the SLV model.
        train: if True, adds a Gaussian noise to the volatility surface.
        only_inputs: if True, generate and return only the volatility grid and X.   
    """

    N = kappa.shape[0]

    # Adjust vector dimensions for correct broadcasting
    kappa_vec = kappa.view(N, 1)
    delta_vec = delta.view(N, 1)
    rho_vec = rho.view(N, 1)
    if isinstance(epsilon, torch.Tensor) and epsilon.ndim == 1:
        epsilon_vec = epsilon.view(N, 1, 1, 1)
    else:
        epsilon_vec = epsilon

    sqrt_1mrho2_vec = torch.sqrt(1.0 - rho_vec ** 2)

    sigma_base_expanded = ctx.sigma_base.unsqueeze(0).unsqueeze(0)

    # Distinguish between the training and the evaluation case
    if train:
        noise = torch.randn((N, 1, ctx.steps_x, ctx.steps_t), device=ctx.device)
        sigma_grid = sigma_base_expanded + epsilon_vec * noise
    else:
        sigma_grid = sigma_base_expanded.expand(N, -1, -1, -1)

    sigma_input = sigma_grid ** 2 

    # Treat the case where the payoff is not needed
    if only_inputs:

        # Generate correlated Brownian increments
        Z = torch.randn(N, 2, device=ctx.device)
        Z1 = Z[:, 0:1]  
        Z2 = Z[:, 1:2]  

        WS_T = ctx.sqrt_T * Z1
        Wv_T = ctx.sqrt_T * (rho_vec * Z1 + sqrt_1mrho2_vec * Z2)

        X = torch.cat([WS_T, Wv_T], dim=1)
        return sigma_input, X, None

    # Compute the matrix eta 
    t_row = ctx.t_arr.unsqueeze(0)
    exp_terms = torch.exp(-2 * kappa_vec * t_row)
    eta_matrix = - (delta_vec ** 2) * (1 + exp_terms) / (2 * kappa_vec)

    # Generate correlated Brownian increments
    Z = torch.randn(N, ctx.sampling_freq, 2, device=ctx.device)
    Z1 = Z[:, :, 0]
    Z2 = Z[:, :, 1]

    dWS = ctx.sqrt_dt * Z1
    dWv = ctx.sqrt_dt * (rho_vec * Z1 + sqrt_1mrho2_vec * Z2)

    if isinstance(v0, torch.Tensor) and v0.ndim > 0:
        v = v0.to(ctx.device).clone()
    else:
        v = torch.full((N,), v0, device=ctx.device)
    x = torch.zeros(N, device=ctx.device)
    sum_log_sq = torch.zeros(N, device=ctx.device)

    for i in range(ctx.sampling_freq):
        
        # Select the eta at time t
        eta = eta_matrix[:, i]

        # Compute the normalized x for grid_sample (it has to be in range [-1, 1])
        x_norm = 2 * (x - ctx.xmin) / (ctx.xmax - ctx.xmin) - 1

        t_val = ctx.t_norm_arr[i]
        t_norm = torch.full((N,), t_val, device=ctx.device)

        grid = torch.stack([t_norm, x_norm], dim=-1).view(N, 1, 1, 2)
        sigma_interp = F.grid_sample(sigma_grid, grid, mode='bicubic',
                                     padding_mode='border', align_corners=True).view(N)

        sigma_hat = torch.exp(v) * sigma_interp

        log_ret = (r - 0.5 * sigma_hat ** 2) * ctx.dt + sigma_hat * dWS[:, i]

        dv = kappa_vec.squeeze() * (eta - v) * ctx.dt + delta_vec.squeeze() * dWv[:, i]

        v = v + dv
        x += log_ret
        sum_log_sq += log_ret ** 2

    payoff = 252 * sum_log_sq / ctx.sampling_freq

    WS_T_sum = torch.sum(dWS, dim=1)
    Wv_T_sum = torch.sum(dWv, dim=1)
    X = torch.stack([WS_T_sum, Wv_T_sum], dim=1)

    return sigma_input, X, payoff.unsqueeze(1)

# -----------------------------------Dataset-------------------------------------
class PEMCDataset(IterableDataset):
    """
    Creates the training dataset.
    """
    def __init__(self, num_samples, intervals, dt, T, device, batch_size, r, epsilon, v0,
                 grid_height, grid_width, sim_chunk_size=2048):
        """
        Arguments:
            num_samples: total number of training samples.
            intervals: intervals used for uniform sampling of theta.
            dt: temporal discretization step.
            T: time to maturity of the derivative.
            device: device where the data is stored (CPU/GPU).
            batch_size: size of the training batch.
            r, epsilon, v0: parameters of theta.
            grid_height: number of points in the log-return grid.
            grid_width: number of points in the time grid.
            sim_chunk_size: dimension of each chunk that is simulated together and accumulated to form a batch.
        """
        super(PEMCDataset, self).__init__()
        self.device = device
        self.num_samples = num_samples
        self.intervals = intervals
        self.dt = dt
        self.n_params = len(intervals)
        self.batch_size = batch_size
        self.sim_chunk_size = sim_chunk_size

        self.batches_per_epoch = num_samples // batch_size + (num_samples % batch_size > 0)

        self.r = r
        self.epsilon = epsilon
        self.v0 = v0
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.ctx = SLVContext(T, grid_height, grid_width, dt, device)

    def __iter__(self):
        for batch_idx in range(self.batches_per_epoch):

            # Computation of the batch size in order to manage the last batch, that could be smaller than the previous ones
            target_batch_size = min(self.batch_size, self.num_samples - batch_idx * self.batch_size)

            # Lists to collect the results from chunks
            thetas, Xs, payoffs, sigmas = [], [], [], []

            # Generate batch on-the-fly dividing it in smaller chunks to allow bigger batch dimension
            with torch.no_grad():

                for i in range(0, target_batch_size, self.sim_chunk_size):

                    current_chunk_size = min(self.sim_chunk_size, target_batch_size - i)

                    chunk_theta = torch.zeros((current_chunk_size, self.n_params + 3), device=self.device)

                    for k, (low, high) in enumerate(self.intervals):
                        chunk_theta[:, k].uniform_(low, high)

                    chunk_theta[:, self.n_params] = self.r
                    chunk_theta[:, self.n_params + 1] = self.epsilon
                    chunk_theta[:, self.n_params + 2] = self.v0

                    c_sigma, c_X, c_payoff = opt_slv_train(self.ctx, self.epsilon, self.r, self.v0, chunk_theta[:, 1], chunk_theta[:, 2], chunk_theta[:, 3])

                    thetas.append(chunk_theta)
                    Xs.append(c_X)
                    payoffs.append(c_payoff)
                    sigmas.append(c_sigma)

                # Concatente the results of the chunked simulation into the batch
                batch_data = (torch.cat(thetas), torch.cat(Xs), torch.cat(payoffs), torch.cat(sigmas))

            yield batch_data

class ValidationDataset(IterableDataset):
    """
    Creates the validation dataset.
    """
    def __init__(self, num_samples, intervals, dt, T, device, r, epsilon, v0, grid_height, grid_width, gen_batch_size=4096, yield_batch_size=2048):
        """
        Arguments:
            num_samples: total number of training samples.
            intervals: intervals used for uniform sampling of theta.
            dt: temporal discretization step.
            T: time to maturity of the derivative.
            device: device where the data is stored (CPU/GPU).
            r, epsilon, v0: parameters of theta.
            grid_height: number of points in the log-return grid.
            grid_width: number of points in the time grid.
            gen_batch_size: size of each batch that is simulated together and accumulated to form the dataset.
            yield_batch_size: dimension of each batch that is yielded.
        """
        super(ValidationDataset, self).__init__()

        self.num_samples = num_samples
        self.n_params = len(intervals)
        self.yield_batch_size = yield_batch_size
        self.ctx = SLVContext(T, grid_height, grid_width, dt, device)

        # Lists to collect the results from batches
        theta_list = []
        sigma_list = []
        X_list = []
        payoff_list = []

        # Generate the full dataset dividing it in smaller batches
        with torch.no_grad():
            for i in range(0, num_samples, gen_batch_size):
                current_bs = min(gen_batch_size, num_samples - i)

                chunk_theta = torch.zeros((current_bs, self.n_params + 3), device=device)
                for k, (low, high) in enumerate(intervals):
                    chunk_theta[:, k].uniform_(low, high)

                chunk_theta[:, self.n_params] = r
                chunk_theta[:, self.n_params + 1] = epsilon
                chunk_theta[:, self.n_params + 2] = v0

                c_sigma, c_X, c_payoff = opt_slv_train(self.ctx, epsilon, r, v0, chunk_theta[:,1], chunk_theta[:,2], chunk_theta[:,3])

                theta_list.append(chunk_theta)
                sigma_list.append(c_sigma)
                X_list.append(c_X)
                payoff_list.append(c_payoff)

                torch.cuda.empty_cache()

            # Concatenate all batches into the final tensors
            self.theta = torch.cat(theta_list, dim=0)
            self.sigma = torch.cat(sigma_list, dim=0)
            self.X = torch.cat(X_list, dim=0)
            self.payoff = torch.cat(payoff_list, dim=0)

    def __iter__(self):
            # Yield the validation dataset in batches
            for i in range(0, self.num_samples, self.yield_batch_size):
                end = min(i + self.yield_batch_size, self.num_samples)
                yield (self.theta[i:end], self.X[i:end], self.payoff[i:end], self.sigma[i:end])

# --------------------------------Model------------------------------------------
class PEMCNetwork(nn.Module):
    """
    Initializes the model.
    """
    def __init__(self, grid_height, grid_width, out_channels_1=32, out_channels_2=64, dropout_rate=0.5,
                 vec_enc_input_dim=9, vec_enc_hidden_dim=512, vec_enc_output_dim=128, dim_embedding=128,
                 synthesizer_hidden_dim=128, output_dim=1):
        """
        Arguments:
            grid_height: number of points in the log-return grid.
            grid_width: number of points in the time grid.
            out_channels_1: number of output channels of the first convolutional layer in the CNN branch.
            out_channels_2: number of output channels of the second convolutional layer in the CNN branch.
            dropout_rate: dropout rate.
            vec_enc_input_dim: input dimension of the vector features encoder.
            vec_enc_hidden_dim: number of neurons in each hidden layer of the vector features encoder.
            vec_enc_output_dim: output dimension of the vector features encoder.
            dim_embedding: output dimension of the CNN branch.
            synthesizer_hidden_dim: number of neurons in each hidden layer of the synthesizer.
            output_dim: output dimension of the synthesizer.
        """
        super(PEMCNetwork, self).__init__()
        
        # CNN branch (VGG-style)
        self.cnn_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels_1,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten()
        )

        # Calculate flattened size after CNN and MaxPool
        flattened_height = grid_height // 2
        flattened_width = grid_width // 2
        flattened_size = flattened_height * flattened_width * out_channels_2
        self.cnn_fc = nn.Linear(flattened_size, dim_embedding)

        # Vector features encoder
        self.vector_encoder = nn.Sequential(
            nn.Linear(vec_enc_input_dim, vec_enc_hidden_dim),
            nn.BatchNorm1d(vec_enc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(vec_enc_hidden_dim, vec_enc_output_dim)
        )

        synth_input_dim = dim_embedding + vec_enc_output_dim

        # Synthesizer
        self.synthesizer = nn.Sequential(
            nn.Linear(synth_input_dim, synthesizer_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(synthesizer_hidden_dim, output_dim)
        )

        self.apply(self._init_weights)

    def forward(self, theta, sigma, x):

        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(1)

        if x.dim() == 1:
            x = x.unsqueeze(1)

        x_vec = torch.cat([theta, x], dim=1)

        # CNN branch
        out_cnn = self.cnn_block(sigma)
        out_cnn = self.cnn_fc(out_cnn)

        # Vector features encoder
        out_vec = self.vector_encoder(x_vec)

        combined = torch.cat((out_cnn, out_vec), dim=1)

        # Synthesizer
        out = self.synthesizer(combined)

        return out

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # Initialize weights
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            
            # Initialize all bias values to zero
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# -------------------------Training and evaluation-------------------------------
class training:
    """
    Trains the model.
    """
    def __init__(self, model, Ntrain, batch_size, intervals, dt, T, r, epsilon, v0, grid_height, grid_width, lr=1e-3):
        """
        Arguments:
            model: "PEMCNetwork" object that represents the model used for training.
            Ntrain: total number of training samples.
            batch_size: size of the training batch.
            intervals: intervals used for uniform sampling of theta.
            dt: temporal discretization step.
            T: time to maturity of the derivative.
            r, epsilon, v0: parameters of theta.
            grid_height: number of points in the log-return grid.
            grid_width: number of points in the time grid.
            lr: learning rate.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Store all the useful parameters
        self.Ntrain = Ntrain
        self.batch_size = batch_size
        self.intervals = intervals
        self.dt = dt
        self.grid_height = grid_height
        self.grid_width = grid_width

        # Model and training setup
        self.model = model.to(self.device).float()
        self.model = torch.compile(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.criterion = nn.MSELoss()

        # Early-stopping variables
        self.best_mare = float('inf')
        self.loss_at_best_mare = float('inf')
        self.best_model_state = None

        # Initialize the training dataset and the DataLoader
        self.train_dataset = PEMCDataset(Ntrain, intervals, dt, T, self.device, batch_size, r, epsilon, v0, grid_height, grid_width)
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
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for theta_val, x_val, y_val, sigma_val in val_loader:
                    output = self.model(theta_val, sigma_val, x_val)

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
        scaler = torch.amp.GradScaler('cuda')

        # Training loop
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            running_loss, total_train_samples = 0, 0

            for theta, x, y, sigma in self.train_loader:

                # Create a batch of the dataset and train the model on it
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output = self.model(theta, sigma, x)
                    loss = self.criterion(output, y)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

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
    def __init__(self, T, dt, grid_height, grid_width):
        """
        Arguments:
            T: time to maturity of the derivative.
            dt: temporal discretization step.
            grid_height: number of points in the log-return grid.
            grid_width: number of points in the time grid.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dt = dt
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.ctx = SLVContext(T, grid_height, grid_width, dt, self.device)

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
                _, _, payoff = opt_slv_eval(self.ctx, batch_theta[:, 5], batch_theta[:, 4], batch_theta[:, 6], batch_theta[:, 1], batch_theta[:, 2], batch_theta[:, 3])
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
                sigma_n, x_n, f = opt_slv_eval(self.ctx, batch_theta_n[:, 5], batch_theta_n[:, 4], batch_theta_n[:, 6], batch_theta_n[:, 1], batch_theta_n[:, 2], batch_theta_n[:, 3])
                g = model(batch_theta_n, sigma_n, x_n)
                sum_diff += torch.sum(f - g)

            # Generate N i.i.d. samples of X
            for batch_idx in range(batches_per_epoch_N):
                current_batch_size_N = min(batch_size, N - batch_idx * batch_size)
                batch_theta_N = theta_tensor[:current_batch_size_N]
                sigma_N, x_N, _ = opt_slv_inputs(self.ctx, batch_theta_N[:, 5], batch_theta_N[:, 4], batch_theta_N[:, 6], batch_theta_N[:, 1], batch_theta_N[:, 2], batch_theta_N[:, 3])
                g_tilda = model(batch_theta_N, sigma_N, x_N)
                sum_g_tilda += torch.sum(g_tilda)

            # Compute PEMC estimator
            PEMC = sum_diff / n + sum_g_tilda / N
        return PEMC.item()


# ----------------------------Optuna Optimization--------------------------------
Ntrain = 3 * 10 ** 6
sampling_freq = 252
intervals = [(50, 150), (1.5, 4.5), (0.1, 1.0), (-0.9, -0.2)] #(S0,k,delta,rho)
T = 1
dt = 1 / sampling_freq
r = 0.02
epsilon = 0.02
v0 = 0
grid_height = 32
grid_width = 32

# Compile simulation functions
slv_train_fn = partial(simulate_slv, train=True, only_inputs=False)
opt_slv_train = torch.compile(slv_train_fn)
slv_eval_fn = partial(simulate_slv, train=False, only_inputs=False)
opt_slv_eval = torch.compile(slv_eval_fn)
slv_inputs_fn = partial(simulate_slv, train=False, only_inputs=True)
opt_slv_inputs = torch.compile(slv_inputs_fn)

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
    hyperparameters_val_set = ValidationDataset(val_dim, intervals, dt, T, device, r, epsilon, v0, grid_height, grid_width)
    hyperparameters_loader = DataLoader(hyperparameters_val_set, batch_size=None)

    def objective(trial):
        model = None
        trainer = None
        try:
            batch_size = trial.suggest_categorical('batch_size', [16384, 32768])
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.1)
            out_channels_1 = trial.suggest_categorical('out_channels_1', [32, 64])
            out_channels_2 = trial.suggest_categorical('out_channels_2', [32, 64])
            synthesizer_hidden_dim = trial.suggest_categorical('synthesizer_hidden_dim', [256, 512, 1024])
            lr = trial.suggest_float('lr', 5e-5, 1e-3, log=True)
            dim_embedding = trial.suggest_categorical('dim_embedding', [256, 512])

            # Create the model
            model = PEMCNetwork(grid_height=grid_height, grid_width=grid_width, dropout_rate=dropout_rate,
                                out_channels_1=out_channels_1, out_channels_2=out_channels_2, dim_embedding=dim_embedding,
                                synthesizer_hidden_dim=synthesizer_hidden_dim)

            trainer = training(model, Ntrain, batch_size, intervals, dt, T, r, epsilon, v0, grid_height, grid_width, lr=lr)
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
    model = PEMCNetwork(grid_height=grid_height, grid_width=grid_width, dropout_rate=best_params['dropout_rate'],
                        out_channels_1=best_params['out_channels_1'], out_channels_2=best_params['out_channels_2'],
                        dim_embedding=best_params['dim_embedding'],
                        synthesizer_hidden_dim=best_params['synthesizer_hidden_dim'])
    model = model.to(device).float()

    # Upload weights and biases
    state_dict = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state_dict)

    print("Weights loaded successfully")

    # Compile the model
    model = torch.compile(model)
    print("Model compiled successfully")

# Train the model
else:
    # Initialize the validation set for early-stopping
    early_stopping_val_set = ValidationDataset(val_dim, intervals, dt, T, device, r, epsilon, v0, grid_height, grid_width)
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
    model = PEMCNetwork(grid_height=grid_height, grid_width=grid_width, dropout_rate=best_params['dropout_rate'],
                        out_channels_1=best_params['out_channels_1'], out_channels_2=best_params['out_channels_2'],
                        dim_embedding=best_params['dim_embedding'],
                        synthesizer_hidden_dim=best_params['synthesizer_hidden_dim'])
    trainer = training(model, Ntrain, best_params['batch_size'], intervals, dt, T, r, epsilon, v0, grid_height, grid_width, lr=best_params['lr'])
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
theta_eval = [100, 3.0, 0.5, -0.5, 0.02, 0, 0] #(S0,k,delta,rho,r,epsilon,v0)
batch_eval = 32768

# Set the seed for evaluation
set_all_seeds(42)

evaluator = evaluation(T, dt, grid_height, grid_width)
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
