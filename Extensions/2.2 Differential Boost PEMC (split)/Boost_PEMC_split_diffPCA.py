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
GT_FILE = os.path.join(BASE_DIR, "Boost_PEMC_ground_truth.json")
PARAMS_FILE = os.path.join(BASE_DIR, "Boost_PEMC_best_params.json")
MODEL_FILE = os.path.join(BASE_DIR, "Boost_PEMC_trained_model.pth")

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
    Applies the "split" data transformation step of the Differential PEMC.
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
        
        self.y_mean = None
        self.y_std = None
        self.mu_theta = None
        
        self.P2_theta = None       
        self.D2_inv_theta = None   
        self.P3_theta = None      
        self.n_pca_theta = None
        self.n_diff_theta = None

        self.P2_w = None
        self.D2_inv_w = None
        self.P3_w = None
        self.n_pca_W = None
        self.n_diff_W = None

    def compute_diffPCA(self, X, Z):
        """
        Applies PCA and differential PCA to the input data. 

        Arguments:
            X: centered input matrix.
            Z: scaled gradients matrix.
        """
        n_samples, n_features = X.shape
        
        # PCA
        # Perform eigenvalue decomposition of X^T * X / m 
        Cov_X = (X.T @ X) / n_samples
        d2, P2 = torch.linalg.eigh(Cov_X)
        
        # Order descending
        d2 = torch.flip(d2, dims=[0])
        P2 = torch.flip(P2, dims=[1])
        
        # Compute the number of components to keep
        sumd2 = torch.cumsum(d2, dim=0)
        total_variance_d2 = sumd2[-1]

        if self.n_components_pca is not None:
            if self.n_components_pca >= 1:
                n_comp_pca = int(self.n_components_pca)
            else:
                sumd2_ratio = sumd2 / total_variance_d2
                target_val = torch.tensor(self.n_components_pca, device=d2.device)
                n_comp_pca = torch.searchsorted(sumd2_ratio, target_val).item() + 1
        else:
            n_comp_pca = min(n_samples, n_features)
        
        d2 = d2[:n_comp_pca]
        P2 = P2[:, :n_comp_pca]
        
        # Compute scaling matrices 
        d2_inv_sqrt = torch.diag(1.0 / torch.sqrt(d2))
        d2_sqrt = torch.diag(torch.sqrt(d2))
        
        # Update differentials
        Z2 = (Z @ P2) @ d2_sqrt
        
        # Differential PCA
        # Perform eigenvalue decomposition of X_bar_2^T * X_bar_2 / m       
        Cov_Z = (Z2.T @ Z2) / n_samples
        d3, P3 = torch.linalg.eigh(Cov_Z)
        
        d3 = torch.flip(d3, dims=[0])
        P3 = torch.flip(P3, dims=[1])
        
        # Compute the number of components to keep
        sumd3 = torch.cumsum(d3, dim=0)
        total_variance_d3 = sumd3[-1]

        if self.n_components_diff_pca is not None:
            if self.n_components_diff_pca >= 1:
                n_comp_diff = int(self.n_components_diff_pca)
            else:
                sumd3_ratio = sumd3 / total_variance_d3
                target_val = torch.tensor(self.n_components_diff_pca, device=d3.device)
                n_comp_diff = torch.searchsorted(sumd3_ratio, target_val).item() + 1
        else:
            n_comp_diff = min(n_samples, n_features)
        
        P3 = P3[:, :n_comp_diff]
        
        return P2, d2_inv_sqrt, P3, n_comp_pca, n_comp_diff

    def fit(self, theta, W, y, grads_theta, grads_W, intervals=None):
        """
        Fits the transformation matrices and the scaling tensors.

        Arguments:
            theta: theta parameter.
            W: Brownian motion increments.
            y: label.
            grads_theta: gradients of the label with respect to theta. 
            grads_W: gradients of the label with respect to W.
            intervals: sampling intervals of the components of theta.
        """
        # Center the inputs
        if intervals is not None:
            low = torch.tensor([i[0] for i in intervals], device=self.device)
            high = torch.tensor([i[1] for i in intervals], device=self.device)
            self.mu_theta = (low + high) / 2.0
        else:
            self.mu_theta = theta.mean(dim=0)
            
        X1_theta = theta - self.mu_theta
        X1_w = W 
        
        self.y_mean = y.mean()
        self.y_std = y.std()
        
        # Scale gradients 
        Z1_theta = grads_theta / self.y_std
        Z1_w = grads_W / self.y_std
        
        # Apply PCA and differential PCA to theta
        self.P2_theta, self.D2_inv_theta, self.P3_theta, self.n_pca_theta, self.n_diff_theta = self.compute_diffPCA(X1_theta, Z1_theta)
        
        # Apply PCA and differential PCA to W
        self.P2_w, self.D2_inv_w, self.P3_w, self.n_pca_W, self.n_diff_W = self.compute_diffPCA(X1_w, Z1_w)
        
        print(f"Theta: dim: {theta.shape[1]} -> PCA: {self.n_pca_theta} -> DiffPCA: {self.n_diff_theta}")
        print(f"\nX: dim: {W.shape[1]} -> PCA: {self.n_pca_W} -> DiffPCA: {self.n_diff_W}")

    def transform(self, theta, W, y=None):
        """
        Transforms the inputs and the label using the fitted tensors.

        Arguments:
            theta: theta parameter.
            W: Brownian motion increments.
            y: label.
        """
        # Theta pipeline
        t_1 = theta - self.mu_theta

        t_2 = (t_1 @ self.P2_theta) @ self.D2_inv_theta

        t_3 = t_2 @ self.P3_theta
        
        # W pipeline
        w_2 = (W @ self.P2_w) @ self.D2_inv_w
        w_3 = w_2 @ self.P3_w
        
        if y is None:
            return t_3, w_3
        else:
            # Return normalized label
            return t_3, w_3, (y - self.y_mean) / self.y_std

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

    payoff_aritm = simulate_arithmetic_asian_option_payoff(N_calibration, sampling_freq, dt, W_dt, theta, device)
    payoff_geom = simulate_geometric_asian_option_payoff(N_calibration, sampling_freq, dt, W_dt, theta, device)
    label = payoff_aritm - payoff_geom

    # Compute gradients of the label with respect to theta and W
    grads_raw = torch.autograd.grad(outputs=label, inputs=[theta, W_dt], grad_outputs=torch.ones_like(label))
    grads_theta = grads_raw[0]
    grads_W_dt = grads_raw[1]

    # Initialize and fit the transformer
    transformer = DiffPCA(n_components_pca=1-1e-10, n_components_diff_pca=1-1e-2, device=device)
    transformer.fit(theta.detach(), W_dt.detach(), label.detach(), grads_theta, grads_W_dt, intervals)

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

            payoff_aritm = simulate_arithmetic_asian_option_payoff(current_batch_size, self.sampling_freq, self.dt, W_dt, theta, self.device)
            payoff_geom = simulate_geometric_asian_option_payoff(current_batch_size, self.sampling_freq, self.dt, W_dt, theta, self.device)
            label = payoff_aritm - payoff_geom

            # Transform theta, W_dt and the label
            transformed_theta, transformed_W_dt, scaled_label = self.transformer.transform(theta, W_dt, label)

            yield transformed_theta.detach(), transformed_W_dt.detach(), scaled_label.detach()

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

        # Generate all labels 
        payoff_aritm = simulate_arithmetic_asian_option_payoff(num_samples, sampling_freq, dt, W_dt, theta, device)
        payoff_geom = simulate_geometric_asian_option_payoff(num_samples, sampling_freq, dt, W_dt, theta, device)
        self.label = payoff_aritm - payoff_geom

        self.transformed_theta, self.transformed_W_dt = transformer.transform(theta, W_dt)

    def __iter__(self):
        yield self.transformed_theta.detach(), self.transformed_W_dt.detach(), self.label.detach()

# --------------------------------Model------------------------------------------
class PEMCNetwork(nn.Module):
    """
    Initializes the model.
    """
    def __init__(self, transformed_theta_dim, transformed_W_dt_dim, theta_hidden=256, combined_hidden=256, output_dim=1):
        """
        Arguments:
            transformed_theta_dim: dimension of the transformed theta.
            transformed_W_dt_dim: dimension of the transformed W_dt.
            theta_hidden: number of neurons in each hidden layer of the theta network branch.
            combined_hidden: number of neurons in each hidden layer of the combined network.
            output_dim: dimension of the network's output.
        """
        super(PEMCNetwork, self).__init__()

        # Theta network branch
        self.theta_branch = nn.Sequential(
            nn.Linear(transformed_theta_dim, theta_hidden),
            nn.BatchNorm1d(theta_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(theta_hidden, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # X network branch
        x_hidden = max(32, 2 * transformed_W_dt_dim)
        self.x_branch = nn.Sequential(
            nn.Linear(transformed_W_dt_dim, x_hidden),
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
            theta_val, x_val, y_val_descaled = next(iter(val_loader))

            # Compute the MSE loss to be used for hyperparameter tuning
            output = self.model(theta_val, x_val) * self.transformer.y_std + self.transformer.y_mean
            loss = self.criterion(output, y_val_descaled)

            # Compute the modified MARE loss to be used for early-stopping
            total_samples = theta_val.size(0)
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
    Computes the MC, CV and Boost PEMC estimators.
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

    def evaluate_Boost_PEMC(self, model, N, n, theta, transformer):
        """
        Computes the Boost PEMC estimator.

        Arguments:
            model: "PEMCNetwork" object that represents the model used to compute the Boost PEMC estimator.
            N: N=10n.
            n: sample size.
            theta: vector of the evaluation parameters.
            transformer: object containing the transformation matrices and the scaling tensors.
        """
        # Generate n paired samples (label, features)
        theta_tensor = torch.tensor(theta, device=self.device).repeat(int(n), 1)
        W_dt = torch.normal(0.0, float(np.sqrt(self.dt)), size=(int(n), self.sampling_freq), device=self.device)
        f = simulate_arithmetic_asian_option_payoff(int(n), self.sampling_freq, self.dt, W_dt, theta_tensor, self.device)
        payoff_geom = simulate_geometric_asian_option_payoff(int(n), self.sampling_freq, self.dt, W_dt, theta_tensor, self.device)

        # Generate N samples of theta and W_dt
        theta_tensor_tilda = torch.tensor(theta, device=self.device).repeat(int(N), 1)
        W_dt_tilda = torch.normal(0.0, float(np.sqrt(self.dt)), size=(int(N), self.sampling_freq), device=self.device)

        transformed_theta, transformed_W_dt = transformer.transform(theta_tensor, W_dt, None)
        transformed_theta_tilda, transformed_W_dt_tilda = transformer.transform(theta_tensor_tilda, W_dt_tilda, None)


        expected_payoff_exact = geometric_asian_option_closed_form_expected_payoff(theta[0], theta[1], theta[2], theta[3], self.sampling_freq * self.dt, self.sampling_freq)

        # Set the model to evaluation mode
        model.eval()

        # Run inference
        with torch.no_grad():
            g = model(transformed_theta, transformed_W_dt) * transformer.y_std + transformer.y_mean
            g_tilda = model(transformed_theta_tilda, transformed_W_dt_tilda) * transformer.y_std + transformer.y_mean

        # Compute Boost PEMC estimator
        Boost_PEMC = torch.mean(f - payoff_geom - g) + torch.mean(g_tilda) + expected_payoff_exact

        return Boost_PEMC.item()

# ----------------------------Optuna optimization--------------------------------
# Sampling parameters
Ntrain = 128 * 10 ** 2
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

dim_theta_trans = global_transformer.n_diff_theta
dim_w_trans = global_transformer.n_diff_W
print(f"Network input dims -> Theta: {dim_theta_trans}, W_dt: {dim_w_trans}")

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
            batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048, 4096])
            theta_hidden = trial.suggest_int('theta_hidden', 16, 256)
            combined_hidden = trial.suggest_int('combined_hidden', 16, 256)

            # Create the model
            model = PEMCNetwork(transformed_theta_dim=dim_theta_trans, transformed_W_dt_dim=dim_w_trans, theta_hidden=theta_hidden, combined_hidden=combined_hidden)
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
    best_params = load_best_params(dim_w_trans, PARAMS_FILE)

    # Create the model architecture
    model = PEMCNetwork(transformed_theta_dim=dim_theta_trans, transformed_W_dt_dim=dim_w_trans, theta_hidden=best_params['theta_hidden'], combined_hidden=best_params['combined_hidden'])
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
        best_params = load_best_params(dim_w_trans, PARAMS_FILE)

    # Run Optuna hyperparameter tuning
    else:
        print("Starting Optuna study...")
        best_params = run_optuna_study()
        save_best_params(best_params, dim_w_trans, PARAMS_FILE)

    # Retrain with best hyperparameters
    print("Retraining with best hyperparameters...")
    model = PEMCNetwork(transformed_theta_dim=dim_theta_trans, transformed_W_dt_dim=dim_w_trans, theta_hidden=best_params['theta_hidden'],
                          combined_hidden=best_params['combined_hidden'])
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
rmseCV = np.zeros(len(n_values))
rmseBoost_PEMC = np.zeros(len(n_values))

for i, n in enumerate(n_values):
    print(f"Evaluation with n={n}")

    errCV = 0
    errBoost_PEMC = 0

    for j in range(num_runs):
        current_seed = 42 + (i * 10000) + j
        set_all_seeds(current_seed)
        CV = evaluator.evaluate_CV(n, theta_eval)
        Boost_PEMC = evaluator.evaluate_Boost_PEMC(model, 10 * n, n, theta_eval, global_transformer)

        errCV += (CV - ground_truth) ** 2
        errBoost_PEMC += (Boost_PEMC - ground_truth) ** 2

    # Compute RMSE for current n
    rmseCV[i] = np.sqrt(errCV / num_runs)
    rmseBoost_PEMC[i] = np.sqrt(errBoost_PEMC / num_runs)

# Create a dataframe with the RMSE values for each estimator and value of n
errors = pd.DataFrame(
    data=[rmseBoost_PEMC, rmseCV],
    columns=[f'n={n}' for n in n_values],
    index=['Boost PEMC', 'Geometric CV']
)
print(errors)

# Compute the percentage reduction of Boost PEMC with respect to MC
Boost_PEMC_reduction = np.zeros(len(n_values))
for i, n in enumerate(n_values):
  Boost_PEMC_reduction[i] = (errors[f'n={n}']['Geometric CV'] - errors[f'n={n}']['Boost PEMC']) / errors[f'n={n}']['Geometric CV']

# Create a datafame with the percentage reduction of Boost PEMC with respect to MC
reductions = pd.DataFrame(
    data=[Boost_PEMC_reduction],
    columns=[f'n={n}' for n in n_values],
    index=['Boost PEMC']
)
print(reductions.map(lambda x: f"{x:.3%}"))
