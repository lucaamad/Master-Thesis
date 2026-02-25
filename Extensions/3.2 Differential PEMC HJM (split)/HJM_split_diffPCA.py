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
compute_ground_truth = False  # Set to False to load saved ground truth value instead of computing it

torch.set_default_dtype(torch.float32)

BASE_DIR = os.getcwd()
GT_FILE = os.path.join(BASE_DIR, "HJM_ground_truth.json")
PARAMS_FILE = os.path.join(BASE_DIR, "HJM_best_params.json")
MODEL_FILE = os.path.join(BASE_DIR, "HJM_trained_model.pth")

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
class HJMContext:
    """
    Manages the time grids, indices, and pre-calculated tensors for the 
    Heath-Jarrow-Morton (HJM) framework simulation.
    """
    def __init__(self, tmin, tmax, dt, t_prime_0, dt_swap, n_p, device):
        """
        Arguments:
            tmin: starting time for the maturity grid.
            tmax: end time for the maturiy grid.
            dt: discretization timestep for simulations.
            t_prime_0: starting time of the swap and maturity of the swaption.
            dt_swap: length of each swap payment period.
            n_p: number of swap payment periods.
            device: device where the data is stored (CPU/GPU).
        """
        
        self.device = device
        self.dt = dt
        self.sqrt_dt = math.sqrt(dt)
        self.dt_swap = dt_swap

        # Initialize time grids
        self.N_mat = int(tmax / dt) + 1
        self.grid_T = torch.linspace(tmin, tmax, self.N_mat, device=device)
        
        # Number of steps until the option expiry (t_prime_0)
        self.N_time = int(round(t_prime_0 / dt))

        # Used for splitting Brownian motion paths (Midpoint)
        self.half_idx = self.N_time // 2

        # Calculate (T - t) matrix for all points on the grid
        t_col = self.grid_T[:self.N_time].unsqueeze(1)
        T_row = self.grid_T.unsqueeze(0)
        self.tau = (T_row - t_col).unsqueeze(0)
        
        # Mask to ensure we don't calculate for time past maturity (triangular matrix)
        self.mask_tau = (self.tau >= 0).float()

        # Compute noise
        self.std_f = (1.0 / (100.0 * (self.grid_T + 5.0))).view(1, 1, -1)
        self.T_vector = (self.grid_T + 5.0).view(1, 1, -1)

        steps_per_year = int(dt_swap / dt)

        # Indices corresponding to Swap payment dates t'_1 ... t'_{n_p}
        # Shifted by N_time because pricing starts from option expiry
        self.R_indices = (torch.arange(1, n_p + 1, device=device) * steps_per_year + self.N_time).long()
        self.denom_time = (n_p * dt_swap)

        # Payment indices relative to the start of the pricing slice
        self.pay_indices = (torch.arange(1, n_p + 1, device=device) * steps_per_year).long()


def simulate_hjm_swaption(ctx, sigma0_vec, alpha_sigma_vec, f0_vec, c_f_vec, alpha_f_vec, n_paths, C, dW=None, train=True, only_inputs=False, only_payoff=False):
    """
    Simulates the HJM model to price a swaption.
    
    Arguments:  
        ctx: HJMContext object.
        sigma0_vec, alpha_sigma_vec, f0_vec, c_f_vec, alpha_f_vec: parameters of the base volatility surface and the intitial forward curve.
        n_paths: number of paths to generate.
        C: notional amount.
        dW: pre-computed tensor of Brownian motion increments.
        train: if True, adds a Gaussian noise to the base volatility surface and the initial forward curve.
        only_inputs: if True, generate and return only the volatility grid, the initial forward curve and X.   
        only_payoff: if True, return only the payoff.
    """

    # Reshape inputs for broadcasting
    sigma0 = sigma0_vec.view(n_paths, 1, 1)
    alpha_sigma = alpha_sigma_vec.view(n_paths, 1, 1)
    f0 = f0_vec.view(n_paths, 1)
    c_f = c_f_vec.view(n_paths, 1)
    alpha_f = alpha_f_vec.view(n_paths, 1)
    T_grid_view = ctx.grid_T.unsqueeze(0)

    # Compute the base volatility surface and the initial forward curve in (3.16)-(3.17)
    f_det_curve = f0 + c_f * (1.0 - torch.exp(-alpha_f * T_grid_view))
    sigma_det_base = sigma0 * torch.exp(-alpha_sigma * ctx.tau) * ctx.mask_tau

    # If train=True, add a Gaussian noise to the base volatility surface and the initial forward curve
    if train:
        noise_f = torch.randn((n_paths, ctx.N_mat), device=ctx.device) * ctx.std_f.squeeze(1)
        f0_batch = f_det_curve + noise_f
        std_sigma_vec = sigma0 / (2.0 * ctx.T_vector)
        noise_sigma = torch.randn((n_paths, 1, ctx.N_mat), device=ctx.device) * std_sigma_vec
        sigma_total = (sigma_det_base + noise_sigma) * ctx.mask_tau
    else:
        f0_batch = f_det_curve
        sigma_total = sigma_det_base
    
    if dW is None:
        # Generate Brownian Motion increments (dW)
        dW = torch.randn((n_paths, ctx.N_time, 1), device=ctx.device) * ctx.sqrt_dt

    # If only_inputs=True, generate and return only the volatility grid, the initial forward curve and dW   
    if only_inputs:
        grid_dim = 64
        f0_input = F.interpolate(f0_batch.unsqueeze(1), size=grid_dim, mode='linear', align_corners=False).squeeze(1)
        sigma_input = F.interpolate(sigma_total.unsqueeze(1), size=(grid_dim, grid_dim), mode='bilinear',
                                    align_corners=False).squeeze(1)
        return sigma_input, f0_input, dW

    # Discretization of the dynamics of the forward rates curve

    # Integrate sigma over T
    integral_sigma = torch.cumsum(sigma_total, dim=2) * ctx.dt

    # Use the trapezoidal rule to compute the drift term in (3.15)
    sum_drift = torch.sum(sigma_total * integral_sigma, dim=1)
    term_start = sigma_total[:, 0, :] * integral_sigma[:, 0, :]
    term_end = sigma_total[:, -1, :] * integral_sigma[:, -1, :]
    correction = 0.5 * (term_start + term_end)
    drift_term = (sum_drift - correction) * ctx.dt

    # Compute the diffusion term
    diffusion_term = torch.sum(sigma_total * dW, dim=1)
    
    # Evolution of the forward rates in (3.14)
    f_t5 = f0_batch + drift_term + diffusion_term

    # Pricing of the swaption

    # Compute R
    f0_at_dates = f0_batch[:, ctx.R_indices]
    sum_f0 = torch.sum(f0_at_dates, dim=1)
    R_batch = torch.exp(-sum_f0 / ctx.denom_time)

    # Slice the forward rates curve from t_prime_0 onwards
    f_pricing = f_t5[:, ctx.N_time:]
    
    # Integrate forward rates to get the bond prices 
    integral_f = torch.cumsum(f_pricing, dim=1) * ctx.dt

    # In (3.18) the summation ends at j-1
    target_indices = ctx.pay_indices - 1

    # Computation of (3.18)
    B_pay = torch.exp(-integral_f)[:, target_indices]

    # Compute the payoff in (3.12) 
    sum_B = torch.sum(B_pay, dim=1)
    B_end = B_pay[:, -1]
    val = C * ((1.0 - R_batch) * sum_B * ctx.dt_swap + B_end - 1.0)
    payoff = torch.clamp(val, min=0.0).unsqueeze(1)

    # If only_payoff=True, return just the payoff avoiding the interpolation of sigma and f0
    if only_payoff:
        return payoff

    grid_dim = 64
    f0_input = F.interpolate(f0_batch.unsqueeze(1), size=grid_dim, mode='linear', align_corners=False).squeeze(1)
    sigma_input = F.interpolate(sigma_total.unsqueeze(1), size=(grid_dim, grid_dim), mode='bilinear',
                                align_corners=False).squeeze(1)

    return sigma_input, f0_input, dW, payoff

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
        
def setup_global_pca(N_calibration, intervals, T, device, C, t0, dt_swap, n_p, dt_grid, gen_batch_size=1024):
    """
    Fits the transformation matrices and the scaling tensors once for all.

    Arguments:
        N_calibration: total number of training samples.
        intervals: intervals used for uniform sampling of theta.
        T: time to maturity of the derivative.
        device: device where the data is stored (CPU/GPU).
        C: notional amount.
        t0: starting time of the swap and maturity of the swaption.
        dt_swap: length of each swap payment period.
        n_p: number of swap payment periods. 
        dt_grid: temporal discretization step.
        gen_batch_size: size of the data generation batch.
    """    

    print("Initializing Global PCA Transformer...")
    
    ctx = HJMContext(0, T, dt_grid, t0, dt_swap, n_p, device)

    # Lists to collect results
    theta_list, W_dt_list, payoff_list = [], [], []
    grads_list = []

    for i in range(0, N_calibration, gen_batch_size):
        current_bs = min(gen_batch_size, N_calibration - i)
        
        # Generate batch of theta, W and the payoff
        chunk_theta = torch.zeros((current_bs, len(intervals)), device=device)
        for k, (low, high) in enumerate(intervals):
            chunk_theta[:, k].uniform_(low, high)
        
        c_W_dt = torch.randn((current_bs, ctx.N_time, 1), device=device) * ctx.sqrt_dt
        
        # Enable gradient tracking for the batch of theta and W
        chunk_theta.requires_grad_(True)
        c_W_dt.requires_grad_(True)
        
        _, _, _, c_payoff = opt_hjm_train(ctx, chunk_theta[:, 0], chunk_theta[:, 1], chunk_theta[:, 2], chunk_theta[:, 3], chunk_theta[:, 4], current_bs, C, c_W_dt)

        # Compute gradients of the label with respect to theta and W
        grads_raw = torch.autograd.grad(outputs=c_payoff, inputs=[chunk_theta, c_W_dt], grad_outputs=torch.ones_like(c_payoff), retain_graph=False)
        
        # Store detached results
        theta_list.append(chunk_theta.detach())
        W_dt_list.append(c_W_dt.detach())
        payoff_list.append(c_payoff.detach())
        
        batch_grads = torch.cat([g.reshape(current_bs, -1) for g in grads_raw], dim=1)
        grads_list.append(batch_grads.detach())

        # Clean cache
        del chunk_theta, c_W_dt, c_payoff, grads_raw
        torch.cuda.empty_cache()

    # Concatenate all batches
    theta = torch.cat(theta_list, dim=0)
    W_dt = torch.cat(W_dt_list, dim=0)
    payoff = torch.cat(payoff_list, dim=0)
    
    grads = torch.cat(grads_list, dim=0)

    # Initialize and fit the transformer
    transformer = DiffPCA(n_components_pca=1-1e-10, n_components_diff_pca=1-1e-4, device=device)
    transformer.fit(theta, W_dt, payoff, grads, intervals)

    return transformer

# -----------------------------------Dataset-------------------------------------
class PEMCDataset(IterableDataset):
    """
    Creates the training dataset.
    """
    def __init__(self, num_samples, intervals, T, device, batch_size, C, t0, dt_swap, n_p, dt_grid, transformer, sim_chunk_size=2048):
        """
        Arguments:
            num_samples: total number of training samples.
            intervals: intervals used for uniform sampling of theta.
            T: time to maturity of the derivative.
            device: device where the data is stored (CPU/GPU).
            batch_size: size of the training batch.
            C: notional amount.
            t0: starting time of the swap and maturity of the swaption.
            dt_swap: length of each swap payment period.
            n_p: number of swap payment periods. 
            dt_grid: temporal discretization step.
            transformer: object containing the transformation matrices and the scaling tensors.
            sim_chunk_size: dimension of each chunk that is simulated together and accumulated to form a batch.
        """
        super(PEMCDataset, self).__init__()
        self.device = device
        self.num_samples = num_samples
        self.intervals = intervals
        self.n_params = len(intervals)
        self.batch_size = batch_size
        self.C = C
        self.t0 = t0
        self.dt_swap = dt_swap
        self.n_p = n_p
        self.dt_grid = dt_grid
        self.sim_chunk_size = sim_chunk_size
        self.batches_per_epoch = num_samples // batch_size + (num_samples % batch_size > 0)
        self.ctx = HJMContext(0, T, dt_grid, t0, dt_swap, n_p, device)
        self.transformer = transformer

    def __iter__(self):
        for batch_idx in range(self.batches_per_epoch):

            # Computation of the batch size in order to manage the last batch, that could be smaller than the previous ones
            target_batch_size = min(self.batch_size, self.num_samples - batch_idx * self.batch_size)

            # Lists to collect the results from chunks
            thetas, W_dts, sigmas, f0s, payoffs = [], [], [], [], []

            # Generate batch on-the-fly dividing it in smaller chunks to allow bigger batch dimension
            with torch.no_grad():

                for i in range(0, target_batch_size, self.sim_chunk_size):

                    current_chunk_size = min(self.sim_chunk_size, target_batch_size - i)

                    chunk_theta = torch.zeros((current_chunk_size, self.n_params), device=self.device)
                    for k, (low, high) in enumerate(self.intervals):
                        chunk_theta[:, k].uniform_(low, high)

                    c_sigma, c_f0, c_W_dt, c_payoff = opt_hjm_train(self.ctx, chunk_theta[:, 0], chunk_theta[:, 1], chunk_theta[:, 2], chunk_theta[:, 3], chunk_theta[:, 4], current_chunk_size, self.C)

                    thetas.append(chunk_theta)
                    sigmas.append(c_sigma)
                    f0s.append(c_f0)
                    W_dts.append(c_W_dt)
                    payoffs.append(c_payoff)

                theta_batch = torch.cat(thetas, dim=0)
                sigma_batch = torch.cat(sigmas, dim=0)
                f0_batch = torch.cat(f0s, dim=0)
                W_dt_batch = torch.cat(W_dts, dim=0)
                payoff_batch = torch.cat(payoffs, dim=0)

                transformed_features, scaled_label = self.transformer.transform(theta_batch, W_dt_batch, payoff_batch)            
            yield transformed_features.detach(), sigma_batch, f0_batch, scaled_label.detach()

class ValidationDataset(IterableDataset):
    """
    Creates the validation dataset.
    """
    def __init__(self, num_samples, intervals, T, device, C, t0, dt_swap, n_p, dt_grid, transformer, gen_batch_size=4096, yield_batch_size=4096):
        """
        Arguments:
            num_samples: total number of training samples.
            intervals: intervals used for uniform sampling of theta.
            T: time to maturity of the derivative.
            device: device where the data is stored (CPU/GPU).
            C: notional amount.
            t0: starting time of the swap and maturity of the swaption.
            dt_swap: length of each swap payment period.
            n_p: number of swap payment periods. 
            dt_grid: temporal discretization step.
            transformer: object containing the transformation matrices and the scaling tensors.
            gen_batch_size: size of each batch that is simulated together and accumulated to form the dataset.
            yield_batch_size: dimension of each batch that is yielded.
        """
        super(ValidationDataset, self).__init__()

        self.num_samples = num_samples
        self.yield_batch_size = yield_batch_size
        self.n_params = len(intervals)
        self.ctx = HJMContext(0, T, dt_grid, t0, dt_swap, n_p, device)

        # Lists to collect the results from batches
        theta_list = []
        sigma_list = []
        f0_list = []
        W_dt_list = []
        payoff_list = []

        # Generate the full dataset dividing it in smaller batches
        with torch.no_grad():
            for i in range(0, num_samples, gen_batch_size):
                current_bs = min(gen_batch_size, num_samples - i)

                chunk_theta = torch.zeros((current_bs, self.n_params), device=device)
                for k, (low, high) in enumerate(intervals):
                    chunk_theta[:, k].uniform_(low, high)    
                
                c_sigma, c_f0, c_W_dt, c_payoff = opt_hjm_train(self.ctx, chunk_theta[:, 0], chunk_theta[:, 1], chunk_theta[:, 2], chunk_theta[:, 3], chunk_theta[:, 4], current_bs, C)

                theta_list.append(chunk_theta)
                sigma_list.append(c_sigma)
                f0_list.append(c_f0)
                W_dt_list.append(c_W_dt)
                payoff_list.append(c_payoff)

                torch.cuda.empty_cache()

            # Concatenate all batches into the final tensors
            theta = torch.cat(theta_list, dim=0)
            self.sigma = torch.cat(sigma_list, dim=0)
            self.f0 = torch.cat(f0_list, dim=0)
            W_dt = torch.cat(W_dt_list, dim=0)
            self.payoff = torch.cat(payoff_list, dim=0)

            self.transformed_features = transformer.transform(theta, W_dt)

    def __iter__(self):
        # Yield the validation dataset in batches
        for i in range(0, self.num_samples, self.yield_batch_size):
            end = min(i + self.yield_batch_size, self.num_samples)
            yield (self.transformed_features[i:end], self.sigma[i:end], self.f0[i:end], self.payoff[i:end])


# --------------------------------Model------------------------------------------
class FunctionEncoder2D(nn.Module):
    """
    Initializes the 2D function encoder branch.
    """
    def __init__(self, c_in=1, ch1=32, ch2=32):
        """
        Arguments:
            c_in: number of input channels of the first convolutional layer.
            ch1: number of output channels of the first convolutional layer.
            ch2: number of output channels of the second convolutional layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, ch1, kernel_size=(1, 3), stride=(1, 3), padding=0),
            nn.BatchNorm2d(ch1),
            nn.ReLU(),
            nn.Conv2d(ch1, ch2, kernel_size=(1, 3), stride=(1, 3), padding=0),
            nn.BatchNorm2d(ch2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Flatten()
        )

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        return self.net(x)


class FunctionEncoder1D(nn.Module):
    """
    Initializes the 1D function encoder branch.
    """
    def __init__(self, c_in=1, ch1=16, ch2=16):
        """
        Arguments:
            c_in: number of input channels of the first convolutional layer.
            ch1: number of output channels of the first convolutional layer.
            ch2: number of output channels of the second convolutional layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, ch1, kernel_size=10, stride=3, padding=0),
            nn.BatchNorm1d(ch1),
            nn.ReLU(),
            nn.Conv1d(ch1, ch2, kernel_size=10, stride=3, padding=0),
            nn.BatchNorm1d(ch2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2, padding=0),
            nn.Flatten()
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.net(x)


class VectorFeatureEncoder(nn.Module):
    """
    Initializes the vector features encoder branch.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Arguments: 
            input_dim: input dimension.
            hidden_dim: number of neurons in each hidden layer.
            output_dim: output dimension.
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.shortcut_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.relu = nn.ReLU()

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.block1(x)
        # Skip connection and final ReLU
        shortcut = self.shortcut_projection(x)
        out += shortcut
        out = self.relu(out)

        out = self.block2(out)
        return out


class PEMCNetwork(nn.Module):
    """
    Initializes the whole architecture of the model.
    """
    def __init__(self, enc_vec_input_dim, out_channels2d_1, out_channels2d_2, out_channels1d_1, out_channels1d_2, hidden_dim_synthesizer):
        """
        Arguments:
            enc_vec_input_dim: dimension of the concatenated tensor after the data preparation.
            out_channels2d_1: number of output channels of the first convolutional layer of the 2D function encoder.
            out_channels2d_2: number of output channels of the second convolutional layer of the 2D function encoder. 
            out_channels1d_1: number of output channels of the first convolutional layer of the 1D function encoder.
            out_channels1d_2: number of output channels of the second convolutional layer of the 1D function encoder.
            hidden_dim_synthesizer: number of neurons in the hidden layers of the synthesizer
        """
        super().__init__()

        self.dim_2d_side = 64
        self.dim_1d_len = 64
        self.enc_vec_hidden_dim = 512
        self.enc_vec_output_dim = 128

        self.enc_2d = FunctionEncoder2D(ch1=out_channels2d_1, ch2=out_channels2d_2)
        self.enc_1d = FunctionEncoder1D(ch1=out_channels1d_1, ch2=out_channels1d_2)
        self.enc_vec = VectorFeatureEncoder(input_dim=enc_vec_input_dim, hidden_dim=self.enc_vec_hidden_dim, output_dim=self.enc_vec_output_dim)

        # Calculate flattened size after CNN and AvgPool (fomulas in the PyTorch documentation of Conv2d and AvgPool2d)
        h1 = self.dim_2d_side
        w1 = (self.dim_2d_side - 3) // 3 + 1

        h2 = h1
        w2 = (w1 - 3) // 3 + 1

        h_out = h2 // 2
        w_out = w2 // 2
        flat_dim_2d = out_channels2d_2 * h_out * w_out

        # Calculate flattened size after CNN and AvgPool (fomulas in the PyTorch documentation of Conv1d and AvgPool1d)
        l1 = (self.dim_1d_len - 10) // 3 + 1

        l2 = (l1 - 10) // 3 + 1

        l_out = l2 // 2

        flat_dim_1d = out_channels1d_2 * l_out

        flat_dim_vec = self.enc_vec_output_dim

        syn_in = flat_dim_2d + flat_dim_1d + flat_dim_vec + enc_vec_input_dim

        self.synthesizer_in = nn.Sequential(
            nn.Linear(syn_in, hidden_dim_synthesizer),
            nn.BatchNorm1d(hidden_dim_synthesizer)
        )

        self.relu = nn.ReLU()

        self.projection = nn.Sequential(
            nn.Linear(syn_in, hidden_dim_synthesizer),
            nn.BatchNorm1d(hidden_dim_synthesizer)
        )

        self.synthesizer_hidden = nn.Sequential(
            nn.Linear(hidden_dim_synthesizer, hidden_dim_synthesizer),
            nn.BatchNorm1d(hidden_dim_synthesizer)
        )

        self.synthesizer_out = nn.Linear(hidden_dim_synthesizer, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # Initialize weights
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            
            # Initialize all bias values to zero
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, features, sigma, f0):

        f2 = self.enc_2d(sigma)
        f1 = self.enc_1d(f0)
        fv = self.enc_vec(features)

        combined = torch.cat([f2, fv, f1, features], dim=1)
       
        out_1 = self.synthesizer_in(combined) 

        # Skip connection
        shortcut_1 = self.projection(combined)  
        x2 = self.relu(out_1 + shortcut_1)
        
        out_2 = self.synthesizer_hidden(x2)

        # Skip connection
        x3 = self.relu(out_2 + x2)
        
        out = self.synthesizer_out(x3)

        return out

# -------------------------Training------------------------------------------
class training:
    """
    Trains the model.
    """
    def __init__(self, model, Ntrain, batch_size, intervals, T, C, t0, dt_swap, n_p, dt_grid, transformer, lr):
        """
        Arguments:
            model: "PEMCNetwork" object that represents the model used for training.
            Ntrain: total number of training samples.
            batch_size: size of the training batch.
            intervals: intervals used for uniform sampling of theta.
            T: time to maturity of the derivative.
            C: notional amount.
            t0: starting time of the swap and maturity of the swaption.
            dt_swap: length of each swap payment period.
            n_p: number of swap payment periods. 
            dt_grid: temporal discretization step.
            transformer: object containing the transformation matrices and the scaling tensors.
            lr: learning rate.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Store all the useful parameters
        self.Ntrain = Ntrain
        self.batch_size = batch_size
        self.intervals = intervals
        self.T = T
        self.C = C  
        self.t0 = t0
        self.dt_swap = dt_swap
        self.n_p = n_p
        self.dt_grid = dt_grid

        # Model and training setup
        self.model = model.to(self.device).float()
        self.model = torch.compile(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.criterion = nn.MSELoss()

        # Early-stopping variables
        self.best_mare = float('inf')
        self.best_model_state = None

        # Initialize the training dataset and the DataLoader
        self.train_dataset = PEMCDataset(Ntrain, intervals, T, self.device, batch_size, C, t0, dt_swap, n_p, dt_grid, transformer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=None)
        
        self.transformer = transformer

    def validate(self, val_loader):
        """
        Computes MSE and modified MARE on the validation dataset.
        
        Arguments:
            val_loader: DataLoader for the validation set.
        """
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        sum_predictions, sum_targets = 0.0, 0.0

        # Compute the validation losses on the whole validation set
        with torch.no_grad():
            for features_val, sigma_val, f0_val, y_val_descaled in val_loader:
                output = self.model(features_val, sigma_val, f0_val) * self.transformer.y_std + self.transformer.y_mean
                
                loss = self.criterion(output, y_val_descaled)

                batch_size = features_val.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                sum_predictions += output.sum().item()
                sum_targets += y_val_descaled.sum().item()

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

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            running_loss, total_train_samples = 0, 0

            for features, sigma, f0, y in self.train_loader:

                # Create a batch of the dataset and train the model on it
                self.optimizer.zero_grad()

                output = self.model(features, sigma, f0)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                current_bs = features.size(0)
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
    def __init__(self, T, C, t0, dt_swap, n_p, dt_grid):
        """
        Arguments:
            T: time to maturity of the derivative.
            C: notional amount.
            t0: starting time of the swap and maturity of the swaption.
            dt_swap: length of each swap payment period.
            n_p: number of swap payment periods. 
            dt_grid: temporal discretization step.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ctx = HJMContext(0, T, dt_grid, t0, dt_swap, n_p, self.device)
        self.T, self.C, self.t0 = T, C, t0
        self.dt_swap, self.n_p, self.dt_grid = dt_swap, n_p, dt_grid

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
                payoff = opt_hjm_payoff(self.ctx, batch_theta[:, 0], batch_theta[:, 1], batch_theta[:, 2], batch_theta[:, 3], batch_theta[:, 4], current_size, self.C)
                sum_payoffs += torch.sum(payoff)
        return (sum_payoffs / n).item()

    def evaluate_PEMC(self, model, N, n, theta_tensor, batch_size, transformer):
        """
        Computes the PEMC estimator.

        Arguments:
            model: "PEMCNetwork" object that represents the model used to compute the PEMC estimator.
            N: N=10n.
            n: sample size.
            theta_tensor: tensor that contains the evaluation parameters.
            batch_size: size of the batch used to compute the PEMC estimator.
            transformer: object containing the transformation matrices and the scaling tensors.
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
                sigma_n, f0_n, W_dt_n, f = opt_hjm_eval(self.ctx, batch_theta_n[:, 0], batch_theta_n[:, 1], batch_theta_n[:, 2], batch_theta_n[:, 3], batch_theta_n[:, 4], current_batch_size_n, self.C)
                transformed_features_n = transformer.transform(batch_theta_n, W_dt_n)
                g = model(transformed_features_n, sigma_n, f0_n) * transformer.y_std + transformer.y_mean
                sum_diff += torch.sum(f - g)

            # Generate N samples of theta and W_dt
            for batch_idx in range(batches_per_epoch_N):
                current_batch_size_N = min(batch_size, N - batch_idx * batch_size)
                batch_theta_N = theta_tensor[:current_batch_size_N]
                sigma_N, f0_N, W_dt_N = opt_hjm_inputs(self.ctx, batch_theta_N[:, 0], batch_theta_N[:, 1], batch_theta_N[:, 2], batch_theta_N[:, 3], batch_theta_N[:, 4], current_batch_size_N, self.C)
                transformed_features_N = transformer.transform(batch_theta_N, W_dt_N)
                g_tilda = model(transformed_features_N, sigma_N, f0_N) * transformer.y_std + transformer.y_mean
                sum_g_tilda += torch.sum(g_tilda)

            # Compute PEMC estimator
            PEMC = sum_diff / n + sum_g_tilda / N
        return PEMC.item()


# ----------------------------Optuna Optimization--------------------------------
Ntrain = 3 * 10 ** 4
intervals = [(0.01, 0.03), (0.001, 0.9), (0.01, 0.03), (0.01, 0.05), (0.001, 0.9)] #(sigma0,alpha_sigma,f0,c_f,alpha_f)
T = 25
dt_grid = 1 / 52
t0 = 5
dt_swap = 1
C = 100
n_p = 20

# Compile simulation functions
hjm_train_fn = partial(simulate_hjm_swaption, train=True, only_inputs=False, only_payoff=False)
opt_hjm_train = torch.compile(hjm_train_fn)
hjm_eval_fn = partial(simulate_hjm_swaption, train=False, only_inputs=False, only_payoff=False)
opt_hjm_eval = torch.compile(hjm_eval_fn)
hjm_inputs_fn = partial(simulate_hjm_swaption, train=False, only_inputs=True, only_payoff=False)
opt_hjm_inputs = torch.compile(hjm_inputs_fn)
hjm_payoff_fn = partial(simulate_hjm_swaption, train=False, only_inputs=False, only_payoff=True)
opt_hjm_payoff = torch.compile(hjm_payoff_fn)

# Optuna parameters
epochs = 150
patience = 15
n_trials = 100

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"Running on: {torch.cuda.get_device_name(0)}")

# Compute transformation matrices and the sacling tensors on a simulated dataset
global_transformer = setup_global_pca(10000, intervals, T, device, C, t0, dt_swap, n_p, dt_grid)

input_dim = global_transformer.n_diff
print(f"Network input dims : {input_dim}")

# Set the number of samples of the validation set
val_dim = int(Ntrain * 0.1)

def run_optuna_study():

    # Initialize the validation set for the hyperparameter tuning
    hyperparameters_val_set = ValidationDataset(val_dim, intervals, T, device, C, t0, dt_swap, n_p, dt_grid, global_transformer)
    hyperparameters_loader = DataLoader(hyperparameters_val_set, batch_size=None)

    def objective(trial):
        model = None
        trainer = None
        try:
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
            
            lr = trial.suggest_float('lr', 5e-5, 6e-4, log=True)
            
            out_channels2d_1 = trial.suggest_categorical('out_channels2d_1', [4, 8, 16, 32])
            out_channels2d_2 = trial.suggest_categorical('out_channels2d_2', [16, 32, 64])
            
            out_channels1d_1 = trial.suggest_categorical('out_channels1d_1', [4, 8, 16, 32])
            out_channels1d_2 = trial.suggest_categorical('out_channels1d_2', [16, 32, 64])
            
            hidden_dim_synthesizer = trial.suggest_categorical('hidden_dim_synthesizer', [128, 256, 512])

            model = PEMCNetwork(enc_vec_input_dim=input_dim,
                out_channels2d_1=out_channels2d_1,
                out_channels2d_2=out_channels2d_2,
                out_channels1d_1=out_channels1d_1,
                out_channels1d_2=out_channels1d_2,
                hidden_dim_synthesizer=hidden_dim_synthesizer
            )

            trainer = training(model, Ntrain, batch_size, intervals, T, C, t0, dt_swap, n_p, dt_grid, global_transformer, lr=lr)
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
    model = PEMCNetwork(enc_vec_input_dim=input_dim,
        out_channels2d_1=best_params['out_channels2d_1'],
        out_channels2d_2=best_params['out_channels2d_2'],
        out_channels1d_1=best_params['out_channels1d_1'],
        out_channels1d_2=best_params['out_channels1d_2'],
        hidden_dim_synthesizer=best_params['hidden_dim_synthesizer']
    )
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
    early_stopping_val_set = ValidationDataset(val_dim, intervals, T, device, C, t0, dt_swap, n_p, dt_grid, global_transformer)
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
    model = PEMCNetwork(enc_vec_input_dim=input_dim,
        out_channels2d_1=best_params['out_channels2d_1'],
        out_channels2d_2=best_params['out_channels2d_2'],
        out_channels1d_1=best_params['out_channels1d_1'],
        out_channels1d_2=best_params['out_channels1d_2'],
        hidden_dim_synthesizer=best_params['hidden_dim_synthesizer']
    )
    trainer = training(model, Ntrain, best_params['batch_size'], intervals, T, C, t0, dt_swap, n_p, dt_grid, global_transformer,
                       lr=best_params['lr'])
    trainer.fit(num_epochs=epochs, patience=patience, val_loader=early_stopping_loader)

    print(f"Saving trained model to {MODEL_FILE}...")
    torch.save(model.state_dict(), MODEL_FILE)
    print("Model saved successfully")

    # Clean memory
    del early_stopping_val_set, early_stopping_loader
    del trainer.train_dataset
    del trainer.train_loader
    del trainer.optimizer

# --------------------------Metrics evaluation-----------------------------------
# Delete datasets to free memory
if 'hyperparameters_val_set' in globals(): del hyperparameters_val_set
if 'hyperparameters_loader' in globals(): del hyperparameters_loader
if 'study' in globals(): del study
gc.collect()
torch.cuda.empty_cache()

# Evaluation parameters
num_runs = 200
n_values = [1000, 3000, 5000, 7000, 9000, 11000]
theta_eval = [0.015, 0.45, 0.02, 0.03, 0.5] #(sigma0,alpha_sigma,f0,c_f,alpha_f)
batch_eval = 8192

# Set the seed for evaluation
set_all_seeds(42)

evaluator = evaluation(T, C, t0, dt_swap, n_p, dt_grid)
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

    errMC = 0
    errPEMC = 0

    for j in range(num_runs):
        current_seed = 42 + (i * 10000) + j
        set_all_seeds(current_seed)
        MC = evaluator.evaluate_MC(n, theta_tensor, batch_eval)
        PEMC = evaluator.evaluate_PEMC(model, 10 * n, n, theta_tensor, batch_eval, global_transformer)

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
