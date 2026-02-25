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

# ------------------------- Script Settings-------------------------------------
# OPTUNA TUNING
use_saved_params = True  # Set to True to load saved parameters instead of running Optuna

# LOAD MODEL
load_model = True  # Set to True to load directly the retrained model instead of running the retraining

# GROUND TRUTH COMPUTATION
compute_ground_truth = False  # Set to False to load saved ground truth value instead of computing it

torch.set_default_dtype(torch.float64)

BASE_DIR = os.getcwd()
GT_FILE = os.path.join(BASE_DIR, "PEMC_ground_truth.json")
PARAMS_FILE = os.path.join(BASE_DIR, "PEMC_best_params.json")
MODEL_FILE = os.path.join(BASE_DIR, "PEMC_trained_model.pth")

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
    Computes the closed-form expected payoff of a geometric Asian call option with discrete monitoring under the B&S model.

    Arguments:
        r: risk-free rate.
        S0: initial spot price of the underlying asset.
        sigma: volatilty of the underlying asset.
        K: strike price of the geometric Asian option.
        T: maturity of the option.
        n: sampling frequency.
    """
    sigma_n = sigma * np.sqrt((2 * n + 1) * (n + 1) / (6 * n ** 2))
    mu_n = (r - sigma ** 2 / 2) * (n + 1) / (2 * n) + 0.5 * sigma_n ** 2
    d1 = (np.log(S0 / K) + (mu_n + sigma_n ** 2 / 2) * T) / (sigma_n * np.sqrt(T))
    d2 = d1 - sigma_n * np.sqrt(T)
    expected_payoff = (S0 * np.exp(mu_n * T) * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))

    return expected_payoff

# ------------------------------Preprocessing-----------------------------------
class DiffPCA:
    """
    Applies the "full" data transformation step of the Differential PEMC.

    """
    def __init__(self, n_components_pca=1-1e-10, n_components_diff_pca=1-1e-4, device='cuda'):
        """
        Arguments:
            n_components_pca: number of desired PCA components (if >=1) or minimum percentage level of 'variance' (squared magnitude) explained by the desired PCA components (if <1).
            n_components_diff_pca: number of desired differential PCA components (if >=1) or minimum percentage level of 'variance' (squared magnitude) explained by the desired differential PCA components (if <1).
            device: device where the data is stored (CPU/GPU).
        """
        self.n_components_pca = n_components_pca
        self.n_components_diff_pca = n_components_diff_pca
        self.device = device
        
        self.mu_x = None
        self.y_mean = None
        self.y_std = None
        
        self.P2 = None         
        self.d2_inv_sqrt = None 
        self.d2_sqrt = None   
        self.n_pca = None
        
        self.P3 = None         
        self.n_diff = None

    def build_X(self, theta, W_dt):
        """
        Flattens and concatenates the input data.

        Arguments:
            theta: theta parameter.
            W_dt: Brownian motion increments.
        """
        # Flatten W_dt
        W_dt_flat = W_dt.reshape(W_dt.shape[0], -1)
        
        return torch.cat((theta, W_dt_flat), dim=1)

    def compute_pca(self, A, n_components):
        """
        Applies PCA (or differential PCA) to the input data.

        Arguments:
            A: input matrix.
            n_components: number of components to keep (if >=1) or minimum percentage level of 'variance' (or relevance) explained by the components to keep (if <1).
        """
        n_samples, n_features = A.shape
        
        # Perform eigenvalue decomposition of A^T * A / m
        C = A.T @ A / n_samples
        d, P = torch.linalg.eigh(C)

        # Order descending
        d = torch.flip(d, dims=[0])
        P = torch.flip(P, dims=[1])

        # Compute the number of components to keep
        sumd = torch.cumsum(d, dim=0)
        total_variance = sumd[-1]

        if n_components is not None:
            if n_components >= 1:
                n_comp = int(n_components)
            else:
                sumd_ratio = sumd / total_variance
                target_val = torch.tensor(n_components, device=d.device)
                n_comp = torch.searchsorted(sumd_ratio, target_val).item() + 1
        else:
            n_comp = min(n_samples, n_features)
            
        d_reduced = d[:n_comp]
        P_reduced = P[:, :n_comp]
        
        return d_reduced, P_reduced, n_comp

    def fit(self, theta, W_dt, y, grads, intervals=None):
        """
        Fits the transformation matrices and the scaling tensors.

        Arguments:
            theta: theta parameter.
            W_dt: Brownian motion increments.
            y: label.
            grads: gradients of the label with respect to theta and W_dt. 
            intervals: sampling intervals of the components of theta.
        """
        # Concatenate inputs
        X0 = self.build_X(theta, W_dt)

        # Center the concatenated inputs
        self.mu_x = torch.zeros((1, X0.shape[1]), device=self.device)
        if intervals is not None:
             lows = torch.tensor([i[0] for i in intervals], device=self.device)
             highs = torch.tensor([i[1] for i in intervals], device=self.device)
             theta_mid = (highs + lows) / 2.0
             self.mu_x[0, :len(intervals)] = theta_mid

        X1 = X0 - self.mu_x
        
        self.y_mean = torch.mean(y)
        self.y_std = torch.std(y)
        
        # Scale gradients 
        Z1 = grads / self.y_std
        
        # Apply PCA 
        d2, self.P2, self.n_pca = self.compute_pca(X1, self.n_components_pca)
        
        self.d2_inv_sqrt = torch.diag(1.0 / torch.sqrt(d2)) 
        self.d2_sqrt = torch.diag(torch.sqrt(d2))           
        
        # Update differentials
        Z2 = (Z1 @ self.P2) @ self.d2_sqrt
        
        # Apply differential PCA 
        _, self.P3, self.n_diff = self.compute_pca(Z2, self.n_components_diff_pca)
                
        print(f"(Theta, W): dim: {X0.shape[1]} -> PCA: {self.n_pca} -> DiffPCA: {self.n_diff}")

    def transform(self, theta, W_dt, y=None):
        """
        Transforms the inputs and the label using the fitted tensors.

        Arguments:
            theta: theta parameter.
            W_dt: Brownian motion increments.
            y: label.
        """
        # Concatenate inputs
        X0 = self.build_X(theta, W_dt)
        
        # Transformation pipeline
        X1 = X0 - self.mu_x
        
        X2 = (X1 @ self.P2) @ self.d2_inv_sqrt
        
        X3 = X2 @ self.P3
        
        if y is None:
            return X3
        else:
            # Return normalized label
            return X3, (y - self.y_mean) / self.y_std

def setup_global_pca(N_calibration, sampling_freq, intervals, dt, device):
    """
    Fits the transformation matrices and the scaling tensors once for all.

    Arguments:
        N_calibration: total number of training samples.
        sampling_freq: sampling frequency.
        intervals: intervals used for uniform sampling of theta.
        dt: time discretization step.
        device: device where the data is stored (CPU/GPU).
    """
    
    print("Initializing Global PCA Transformer...")

    # Generate theta, W and the payoff
    theta = torch.zeros((N_calibration, len(intervals)), device=device)
    for i, (low, high) in enumerate(intervals):
        theta[:, i].uniform_(low, high)

    W_dt = torch.normal(0.0, float(np.sqrt(dt)), size=(N_calibration, sampling_freq), device=device)

    # Enable gradient tracking for theta and W
    theta.requires_grad_(True)
    W_dt.requires_grad_(True)

    payoff = simulate_arithmetic_asian_option_payoff(N_calibration, sampling_freq, dt, W_dt, theta, device)

    # Compute gradients of the label with respect to theta and W
    grads_raw = torch.autograd.grad(outputs=payoff, inputs=[theta, W_dt], grad_outputs=torch.ones_like(payoff))
    grads = torch.cat(grads_raw, dim=1)

    # Initialize and fit the transformer
    transformer = DiffPCA(n_components_pca=1-1e-10, n_components_diff_pca=1-1e-3, device=device)
    transformer.fit(theta.detach(), W_dt.detach(), payoff.detach(), grads, intervals)

    return transformer

# -----------------------------------Dataset Generation-------------------------------------
class PEMCDataset(IterableDataset):
    """
    Creates the training dataset.
    """
    def __init__(self, num_samples, sampling_freq, intervals, dt, device, batch_size, transformer):
        """
        Arguments:
            num_samples: total number of training samples.
            sampling_freq: sampling frequency.
            intervals: intervals used for uniform sampling of theta.
            dt: time discretization step.
            device: device where the data is stored (CPU/GPU).
            batch_size: size of the training batch.
            transformer: object containing the transformation matrices and the scaling tensors.
        """
        super(PEMCDataset, self).__init__()
        self.device = device
        self.num_samples = num_samples
        self.sampling_freq = sampling_freq
        self.intervals = intervals
        self.dt = dt
        self.n_params = len(intervals)
        self.batch_size = batch_size
        self.batches_per_epoch = self.num_samples // self.batch_size + (self.num_samples % self.batch_size > 0)
        self.transformer = transformer

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
            
            # Transform theta, W_dt and the label
            transformed_features, scaled_label = self.transformer.transform(theta, W_dt, payoff)

            yield transformed_features.detach(), scaled_label.detach()

class ValidationDataset(IterableDataset):
    """
    Creates the validation dataset.
    """
    def __init__(self, num_samples, sampling_freq, intervals, dt, device, transformer):
        """
        Arguments:
            num_samples: total number of training samples.
            sampling_freq: sampling frequency.
            intervals: intervals used for uniform sampling of theta.
            dt: time discretization step.
            device: device where the data is stored (CPU/GPU).
            transformer: object containing the transformation matrices and the scaling tensors.
        """
        super(ValidationDataset, self).__init__()
        self.n_params = len(intervals)

        # Generate all theta values
        theta = torch.zeros((num_samples, self.n_params), device=device)
        for i, (low, high) in enumerate(intervals):
            theta[:, i].uniform_(low, high)

        W_dt = torch.normal(0.0, float(np.sqrt(dt)), size=(num_samples, sampling_freq), device=device)

        # Generate all payoffs 
        self.payoff = simulate_arithmetic_asian_option_payoff(num_samples, sampling_freq, dt, W_dt, theta, device)

        self.transformed_features = transformer.transform(theta, W_dt)

    def __iter__(self):
        yield self.transformed_features.detach(), self.payoff.detach()

# --------------------------------Model------------------------------------------
class PEMCNetwork(nn.Module):
    """
    Initializes the model.
    """
    def __init__(self, transformed_features_dim, combined_hidden=256, output_dim=1):
        """
        Arguments:
            transformed_features_dim: dimension of the concatenated tensor after the data preparation.
            combined_hidden: number of neurons in each hidden layer of the combined network.
            output_dim: dimension of the network's output.
        """
        super(PEMCNetwork, self).__init__()

        self.combined_fc1 = nn.Linear(transformed_features_dim, combined_hidden)
        self.combined_bn1 = nn.BatchNorm1d(combined_hidden)

        self.combined_fc2 = nn.Linear(combined_hidden, combined_hidden)
        self.combined_bn2 = nn.BatchNorm1d(combined_hidden)

        # Skip connection dimension management
        if transformed_features_dim != combined_hidden:
          self.skip_connection = nn.Sequential(
              nn.Linear(transformed_features_dim, combined_hidden),
              nn.BatchNorm1d(combined_hidden)
          )
        else:
          self.skip_connection = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Output layer
        self.output_layer = nn.Linear(combined_hidden, output_dim)

        self.apply(self._init_weights)

    def forward(self, features):

        residual = self.skip_connection(features)

        # First combined layer
        out = self.combined_fc1(features)
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
# -------------------------Training-----------------------------------------------
class training:
    """
    Trains the model.
    """
    def __init__(self, model, Ntrain, batch_size, sampling_freq, intervals, dt, transformer, lr=1e-3):
        """
        Arguments:
            model: "PEMCNetwork" object that represents the model used for training.
            Ntrain: total number of training samples.
            batch_size: size of the training batch.
            sampling_freq: sampling frequency.
            intervals: intervals used for uniform sampling of theta.
            dt: temporal discretization step.
            transformer: object containing the transformation matrices and the scaling tensors.
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

        # Model and training setup
        self.model = model.to(self.device).double()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Early-stopping variables
        self.best_mare = float('inf')
        self.best_model_state = None

        # Initialize the training dataset and the DataLoader
        self.train_dataset = PEMCDataset(Ntrain, sampling_freq, intervals, dt, self.device, batch_size=self.batch_size, transformer=transformer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=None)

        self.transformer = transformer

    def validate(self, val_loader):
        """
        Computes MSE and modified MARE on the validation dataset.

        Arguments:
            val_loader: DataLoader for the validation set.
        """
        self.model.eval()

        # Compute the validation losses on the whole validation set
        with torch.no_grad():
            features_val, y_val_descaled = next(iter(val_loader))

            # Compute the MSE loss to be used for hyperparameter tuning
            output = self.model(features_val) * self.transformer.y_std + self.transformer.y_mean
            loss = self.criterion(output, y_val_descaled)

            # Compute the modified MARE loss to be used for early-stopping
            total_samples = features_val.size(0)
            prediction = output.sum().item()
            target = y_val_descaled.sum().item()
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

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            running_loss, total_train_samples = 0, 0

            for features, y in self.train_loader:

                # Create a batch of the dataset and train the model on it
                self.optimizer.zero_grad()
                output = self.model(features)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                current_bs = features.size(0)
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
    def __init__(self, dt, sampling_freq, intervals):
        """
        Arguments:
            dt: time discretization step.
            sampling_freq: sampling frequency.
            intervals: intervals used for uniform sampling of theta.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dt = dt
        self.sampling_freq = sampling_freq
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
                W_dt = torch.normal(0.0, float(np.sqrt(self.dt)), size=(current_size, self.sampling_freq), device=self.device)
                payoff = simulate_arithmetic_asian_option_payoff(current_size, self.sampling_freq, self.dt, W_dt, theta_tensor[:int(current_size)], self.device)
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
        payoff_aritm = simulate_arithmetic_asian_option_payoff(int(n), self.sampling_freq, self.dt, W_dt, theta_tensor, self.device)
        payoff_geom = simulate_geometric_asian_option_payoff(int(n), self.sampling_freq, self.dt, W_dt, theta_tensor, self.device)
        expected_payoff_exact = geometric_asian_option_closed_form_expected_payoff(theta[0], theta[1], theta[2], theta[3], self.dt * self.sampling_freq, self.sampling_freq)
        cv = torch.mean(payoff_aritm - payoff_geom).item() + expected_payoff_exact

        return cv

    def evaluate_PEMC(self, model, N, n, theta, transformer):
        """
        Computes the PEMC estimator.

        Arguments:
            model: "PEMCNetwork" object that represents the model used to compute the PEMC estimator.
            N: N=10n.
            n: sample size.
            theta: vector of the evaluation parameters.
            transformer: object containing the transformation matrices and the scaling tensors.
        """
        # Generate n paired samples (label, features)
        theta_tensor = torch.tensor(theta, device=self.device).repeat(int(n), 1)
        W_dt = torch.normal(0.0, float(np.sqrt(self.dt)), size=(int(n), self.sampling_freq), device=self.device)
        f = simulate_arithmetic_asian_option_payoff(int(n), self.sampling_freq, self.dt, W_dt, theta_tensor, self.device)

        # Generate N samples of theta and W_dt
        theta_tensor_tilda = torch.tensor(theta, device=self.device).repeat(int(N), 1)
        W_dt_tilda = torch.normal(0.0, float(np.sqrt(self.dt)), size=(int(N), self.sampling_freq), device=self.device)

        transformed_features = transformer.transform(theta_tensor, W_dt, None)
        transformed_features_tilda = transformer.transform(theta_tensor_tilda, W_dt_tilda, None)

        # Set the model to evaluation mode
        model.eval()

        # Run inference
        with torch.no_grad():
            g = model(transformed_features) * transformer.y_std + transformer.y_mean
            g_tilda = model(transformed_features_tilda) * transformer.y_std + transformer.y_mean
       
        # Compute PEMC estimator
        PEMC = torch.mean(f - g) + torch.mean(g_tilda)

        return PEMC.item()

# ----------------------------Optuna optimization--------------------------------
# Sampling parameters
Ntrain = 128
sampling_freq = 252
intervals = [(0.01, 0.03), (80, 120), (0.05, 0.25), (90, 110)]  # (r,S0,sigma,K)
dt = 1 / sampling_freq

# Optuna parameters
epochs = 200
patience = 20
n_trials = 200

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"Running on: {torch.cuda.get_device_name(0)}")

# Compute transformation matrices and the sacling tensors on a simulated dataset
global_transformer = setup_global_pca(10000, sampling_freq, intervals, dt, device)

input_dim = global_transformer.n_diff
print(f"Network input dims : {input_dim}")

# Set the number of samples of the validation set
val_dim = int(Ntrain * 0.1)

def run_optuna_study():

    # Initialize the validation set for the hyperparameter tuning
    hyperparameters_val_set = ValidationDataset(val_dim, sampling_freq, intervals, dt, device, global_transformer)
    hyperparameters_loader = DataLoader(hyperparameters_val_set, batch_size=None)

    def objective(trial):
        model = None
        trainer = None
        try:
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            combined_hidden = trial.suggest_int('combined_hidden', 16, 512)

            # Create the model
            model = PEMCNetwork(input_dim, combined_hidden=combined_hidden)
            trainer = training(model, Ntrain, batch_size, sampling_freq, intervals, dt, global_transformer, lr=1e-3)
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
    best_params = load_best_params(input_dim, PARAMS_FILE)

    # Create the model architecture
    model = PEMCNetwork(input_dim, combined_hidden=best_params['combined_hidden'])
    model = model.to(device).double()

    # Upload weights and biases
    state_dict = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state_dict)

    print("Model loaded successfully")

# Train the model
else:
    # Initialize the validation set for early-stopping
    early_stopping_val_set = ValidationDataset(val_dim, sampling_freq, intervals, dt, device, global_transformer)
    early_stopping_loader = DataLoader(early_stopping_val_set, batch_size=None)

    # Load the best hyperparameters and just do the final retraining
    if use_saved_params:
        print(f"Loading hyperparameters from input...")
        best_params = load_best_params(input_dim, PARAMS_FILE)

    # Run Optuna hyperparameter tuning
    else:
        print("Starting Optuna study...")
        best_params = run_optuna_study()
        save_best_params(best_params, input_dim, PARAMS_FILE)

    # Retrain with best hyperparameters
    print("Retraining with best hyperparameters...")
    model = PEMCNetwork(input_dim, combined_hidden=best_params['combined_hidden'])
    trainer = training(model, Ntrain, best_params['batch_size'], sampling_freq, intervals, dt, global_transformer, lr=1e-3)
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
num_runs = 300
n_values = [1000, 4000, 9000]
theta_eval = [0.02, 100, 0.2, 100]  # (r,S0,sigma,K)
batch_eval = 2048 * 1000

# Set the seed for evaluation
set_all_seeds(42)

evaluator = evaluation(dt, sampling_freq, intervals)
theta_tensor = torch.tensor(theta_eval, device=device).repeat(batch_eval, 1)

# Compute ground truth
if compute_ground_truth:
    print("Computing ground truth...")
    ground_truth = evaluator.evaluate_MC(int(2e9), theta_tensor, batch_eval)
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
rmsePEMC = np.zeros(len(n_values))

for i, n in enumerate(n_values):
    print(f"Evaluation with n={n}")
    errMC = 0
    errCV = 0
    errPEMC = 0

    for j in range(num_runs):
        current_seed = 42 + (i * 10000) + j
        set_all_seeds(current_seed)
        CV = evaluator.evaluate_CV(n, theta_eval)
        PEMC = evaluator.evaluate_PEMC(model, 10 * n, n, theta_eval, global_transformer)
        MC = evaluator.evaluate_MC(n, theta_tensor, batch_eval)

        errMC += (MC - ground_truth) ** 2
        errCV += (CV - ground_truth) ** 2
        errPEMC += (PEMC - ground_truth) ** 2

    # Compute RMSE for current n
    rmseMC[i] = np.sqrt(errMC / num_runs)
    rmseCV[i] = np.sqrt(errCV / num_runs)
    rmsePEMC[i] = np.sqrt(errPEMC / num_runs)

# Create a dataframe with the RMSE values for each estimator and value of n
errors = pd.DataFrame(
    data=[rmseMC, rmsePEMC, rmseCV],
    columns=[f'n={n}' for n in n_values],
    index=['Monte Carlo (MC)', 'PEMC', 'Geometric CV']
)
print(errors)

# Compute the percentage reduction of PEMC with respect to MC
PEMC_reduction = np.zeros(len(n_values))
for i, n in enumerate(n_values):
  PEMC_reduction[i] = (errors[f'n={n}']['Monte Carlo (MC)'] - errors[f'n={n}']['PEMC']) / errors[f'n={n}']['Monte Carlo (MC)']

# Create a datafame with the percentage reduction of PEMC with respect to MC
reductions = pd.DataFrame(
    data=[PEMC_reduction],
    columns=[f'n={n}' for n in n_values],
    index=['PEMC']
)
print(reductions.map(lambda x: f"{x:.3%}"))
