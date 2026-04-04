import random
from pathlib import Path
from unittest import loader

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


set_seed(42)
device = get_device()
print('Device:', device)


def build_mnist_loaders(batch_size=128, train_limit=None, test_limit=None, data_root='data', num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)

    if train_limit is not None:
        train_ds = Subset(train_ds, list(range(min(train_limit, len(train_ds)))))
    if test_limit is not None:
        test_ds = Subset(test_ds, list(range(min(test_limit, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader



import torch

class GaussianDiffusion:
    """
    DDPM (Denoising Diffusion Probabilistic Models) Scheduler.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta scheduler (can be swapped for Cosine for better efficiency)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # alphas_cumprod_prev starts with 1.0 (no noise)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # posterior_variance
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: Add noise to x_0 at step t.
        q(x_t | x_0) = N(x_t; sqrt(alpha_prod)*x_0, (1-alpha_prod)*I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self._get_index(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_prod = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # Get current alpha/beta values
        betas_t = self._get_index(self.betas, t, x.shape)
        alpha_cumprod_t = self._get_index(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_prev_t = self._get_index(self.alphas_cumprod_prev, t, x.shape)
        
        predicted_noise = model(x, t)
        
        # 1. Predict x_0 from the noise
        sqrt_recip_alphas_cumprod = 1.0 / torch.sqrt(alpha_cumprod_t)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alpha_cumprod_t - 1)
        pred_x0 = sqrt_recip_alphas_cumprod * x - sqrt_recipm1_alphas_cumprod * predicted_noise
        
        # 2. Clip x_0 to [-1, 1] to prevent values from blowing up to white
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # 3. Compute posterior mean using the clipped x_0
        posterior_mean_coef1 = self._get_index(self.posterior_mean_coef1, t, x.shape)
        posterior_mean_coef2 = self._get_index(self.posterior_mean_coef2, t, x.shape)
        model_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._get_index(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        

        
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Sample all steps from pure noise to reconstruct an image in latent space.
        """
        model.eval()
        x = torch.randn(shape).to(self.device)
        # Reverse loop from T-1 back to 0
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long).to(self.device)
            x = self.p_sample(model, x, t, i)
        return x

    def _get_index(self, tensor, t, x_shape):
        """Get value at index t and expand to match x_shape."""
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal Position Embedding for time steps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # Handle odd dimension by padding if necessary, but dim should be even
        return emb

class ResnetBlock(nn.Module):
    """
    Residual Block with Time Embedding projection.
    Supports channel dimension changes with short-cut projection.
    """
    def __init__(self, dim, time_emb_dim, out_dim=None):
        super().__init__()
        self.out_dim = out_dim or dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, self.out_dim)
        )
        self.conv1 = nn.Conv2d(dim, self.out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)
        # GroupNorm tends to work better for diffusion than BatchNorm
        self.norm1 = nn.GroupNorm(4, dim)
        self.norm2 = nn.GroupNorm(4, self.out_dim)
        self.act = nn.SiLU()
        
        # Shortcut for residual if dims don't match
        self.shortcut = nn.Conv2d(dim, self.out_dim, 1) if dim != self.out_dim else nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        # Add time embedding
        time_emb = self.mlp(time_emb)
        # Expand time_emb to match spatial dimensions [B, C, 1, 1]
        h = h + time_emb[:, :, None, None]
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class LatentDenoiseNetwork(nn.Module):
    """
    Denoising Network operating on latent tensors of shape [B, latent_channels, H_lat, W_lat].
    For MNIST latents from VAE, shape may be [B, 4, 4, 4].
    """
    def __init__(self, latent_channels=4, model_channels=64, num_res_blocks=3):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        
        self.init_conv = nn.Conv2d(latent_channels, model_channels, 3, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResnetBlock(model_channels, model_channels * 4)
            for _ in range(num_res_blocks)
        ])
        
        self.out_conv = nn.Conv2d(model_channels, latent_channels, 3, padding=1)
        
    def forward(self, x, t):
        # t is shape [B]
        t_emb = self.time_embed(t)
        h = self.init_conv(x)
        for block in self.res_blocks:
            h = block(h, t_emb)
        return self.out_conv(h)


# --- PIXEL UNET ---

class PixelUNet(nn.Module):
    """
    Standard UNet for Diffusion on image space.
    Fits 32x32 ArtBench images.
    """
    def __init__(self, in_channels=3, model_channels=64):
        super().__init__()
        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        
        time_dim = model_channels * 4
        
        # Initial Conv
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Down 1: 28 -> 14
        self.down1_res = ResnetBlock(model_channels, time_dim)
        self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        # Down 2: 14 -> 7
        self.down2_res = ResnetBlock(model_channels, time_dim, out_dim=model_channels * 2)
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        # Middle
        self.mid_res1 = ResnetBlock(model_channels * 2, time_dim)
        self.mid_res2 = ResnetBlock(model_channels * 2, time_dim)
        
        # Up 2: 7 -> 14
        self.up2_conv = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1) # 7 -> 14
        # Skip connection from down2_res is model_channels * 2
        # After concat: model_channels (up) + model_channels*2 (skip) = model_channels * 3
        self.up2_res = ResnetBlock(model_channels * 3, time_dim, out_dim=model_channels)
        
        # Up 1: 14 -> 28
        self.up1_conv = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1) # 14 -> 28
        # Skip connection from down1_res is model_channels
        # After concat: model_channels (up) + model_channels (skip) = model_channels * 2
        self.up1_res = ResnetBlock(model_channels * 2, time_dim, out_dim=model_channels)
        
        # Out
        self.out_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)
        
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        
        # Initial
        h_init = self.init_conv(x) # [B, C, 28, 28]
        
        # Down 1
        h1 = self.down1_res(h_init, t_emb) # [B, C, 28, 28]
        h1_pool = self.down1_pool(h1)      # [B, C, 14, 14]
        
        # Down 2
        h2 = self.down2_res(h1_pool, t_emb) # [B, 2C, 14, 14]
        h2_pool = self.down2_pool(h2)       # [B, 2C, 7, 7]
        
        # Middle
        h_mid = self.mid_res1(h2_pool, t_emb) # [B, 2C, 7, 7]
        h_mid = self.mid_res2(h_mid, t_emb)   # [B, 2C, 7, 7]
        
        # Up 2
        h_up2 = self.up2_conv(h_mid) # [B, C, 14, 14]
        h_up2 = torch.cat([h_up2, h2], dim=1) # [B, 3C, 14, 14]
        h_up2 = self.up2_res(h_up2, t_emb)   # [B, C, 14, 14]
        
        # Up 1
        h_up1 = self.up1_conv(h_up2) # [B, C, 28, 28]
        h_up1 = torch.cat([h_up1, h1], dim=1) # [B, 2C, 28, 28]
        h_up1 = self.up1_res(h_up1, t_emb)   # [B, C, 28, 28]
        
        # Out
        return self.out_conv(h_up1)


class Encoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.net = nn.Sequential(
            # 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x7x7 -> 64x4x4
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Latent projections
        # 64x4x4 -> latent_channels x 4x4 for mu and logvar
        self.mu = nn.Conv2d(64, latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(64, latent_channels, kernel_size=1)
        
    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.initial_conv = nn.Conv2d(latent_channels, 64, kernel_size=1)
        
        self.net = nn.Sequential(
            # 64x4x4 -> 64x7x7
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x7x7 -> 32x14x14
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x14x14 -> 1x28x28
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh() # Output range [-1, 1]
        )
        
    def forward(self, z):
        h = self.initial_conv(z)
        return self.net(h)

class VAE(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


@torch.no_grad()
def sample_diffusion(model, schedule, shape):
    model.eval()
    x = torch.randn(shape, device=device)

    for step in reversed(range(schedule['T'])):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        pred_noise = model(x, t)

        alpha_t = schedule['alphas'][step]
        alpha_bar_t = schedule['alpha_bars'][step]
        beta_t = schedule['betas'][step]

        if step > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            (1.0 / torch.sqrt(alpha_t))
            * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
            + torch.sqrt(beta_t) * noise
        )

    return x


def train_autoencoder(model, loader, epochs=20, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()

    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for x, _ in tqdm(loader, desc=f'AE epoch {epoch + 1}/{epochs}', leave=False):
            x = x.to(device)
            x_hat, _ = model(x)
            loss = 0.5 * F.mse_loss(x_hat, x) + 0.5 * F.l1_loss(x_hat, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            n_batches += 1

        avg = running / max(n_batches, 1)
        history.append(avg)
        print(f'AE epoch {epoch + 1:02d}/{epochs} | recon loss: {avg:.4f}')

    return history


def train_diffusion(model, loader, schedule, epochs=20, lr=2e-4, encode_fn=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()

    for epoch in range(epochs):
        running = 0.0
        n_batches = 0

        for x, _, _ in tqdm(loader, desc=f'Diff epoch {epoch + 1}/{epochs}', leave=False):
            x = x.to(device)
            x = x * 2.0 - 1.0  # Scale images from [0, 1] to [-1, 1] for Gaussian noise
            if encode_fn is not None:
                with torch.no_grad():
                    # For Latent Diffusion: encode pixels to latent space
                    mu, _ = encode_fn(x)
                    x = mu

            # --- TODO START ---
            bs = x.size(0)
            # 1) Sample random diffusion steps
            t = torch.randint(0, schedule.num_timesteps, (bs,), device=device).long()
        
            # 2) Sample target noise
            noise = torch.randn_like(x)
        
            # 3) Get noisy image x_t
            x_t = schedule.q_sample(x_0=x, t=t, noise=noise)
        
            # 4) Predict noise
            predicted_noise = model(x_t, t)
        
            # 5) Compute MSE loss
            loss = F.mse_loss(predicted_noise, noise)

            # 6) Optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            n_batches += 1
            # --- TODO END ---

    avg = running / max(n_batches, 1)
    history.append(avg)
    print(f'Diff epoch {epoch + 1:02d}/{epochs} | loss: {avg:.4f}')

    return history