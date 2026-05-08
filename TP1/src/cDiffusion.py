import argparse
import copy
import json
import os
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
from pathlib import Path
from torchvision.utils import save_image

# data loading and evaluation helpers
from data_loader import (
    load_artbench_train_split,
    load_artbench_splits,
    build_transform,
    HFDatasetTorch,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    setup_artbench_from_csv_subset,
)
from evaluation import (
    EVAL_CONFIG,
    base_evaluation,
    build_feature_extractor,
    extract_inception_features,
    feature_statistics,
    frechet_distance,
    get_real_samples,
    kid_score,
)

try:
    import optuna
except ImportError:
    optuna = None




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
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu', beta_schedule='linear'):
        self.num_timesteps = num_timesteps
        self.device = device

        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps).to(device)
        else:
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

    @staticmethod
    def _cosine_beta_schedule(num_timesteps, s=0.008):
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 1e-4, 0.999)

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
    def p_sample(self, model, x, t, t_index, labels=None, guidance_scale=1.0):
        # Get current alpha/beta values
        betas_t = self._get_index(self.betas, t, x.shape)
        alpha_cumprod_t = self._get_index(self.alphas_cumprod, t, x.shape)
        alpha_cumprod_prev_t = self._get_index(self.alphas_cumprod_prev, t, x.shape)
        
        if labels is None:
            predicted_noise = model(x, t, None)
        elif guidance_scale > 1.0:
            cond_pred = model(x, t, labels)
            uncond_pred = model(x, t, None)
            predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        else:
            predicted_noise = model(x, t, labels)
        
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
    def p_sample_loop(self, model, shape, labels=None, guidance_scale=1.0, initial_noise=None):
        """
        Sample all steps from pure noise to reconstruct an image in latent space.
        """
        model.eval()
        if initial_noise is not None:
            x = initial_noise.to(self.device).clone()
        else:
            x = torch.randn(shape, device=self.device)
        # Reverse loop from T-1 back to 0
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((shape[0],), i, dtype=torch.long).to(self.device)
            x = self.p_sample(model, x, t, i, labels=labels, guidance_scale=guidance_scale)
        return x

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, labels=None, guidance_scale=1.0, sample_steps=100, eta=0.0, initial_noise=None):
        """DDIM sampling with optional classifier-free guidance."""
        model.eval()
        sample_steps = max(1, min(int(sample_steps), self.num_timesteps))
        if initial_noise is not None:
            x = initial_noise.to(self.device).clone()
        else:
            x = torch.randn(shape, device=self.device)
        step_indices = torch.linspace(self.num_timesteps - 1, 0, sample_steps, device=self.device).long()

        for i, step in enumerate(step_indices):
            t_val = int(step.item())
            t = torch.full((shape[0],), t_val, dtype=torch.long, device=self.device)

            if labels is None:
                predicted_noise = model(x, t, None)
            elif guidance_scale > 1.0:
                cond_pred = model(x, t, labels)
                uncond_pred = model(x, t, None)
                predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                predicted_noise = model(x, t, labels)

            alpha_bar_t = self._get_index(self.alphas_cumprod, t, x.shape)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            if i == len(step_indices) - 1:
                x = pred_x0
                break

            t_prev_val = int(step_indices[i + 1].item())
            t_prev = torch.full((shape[0],), t_prev_val, dtype=torch.long, device=self.device)
            alpha_bar_prev = self._get_index(self.alphas_cumprod, t_prev, x.shape)

            if eta > 0.0:
                sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                sigma = sigma * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
                noise = torch.randn_like(x)
            else:
                sigma = torch.zeros_like(alpha_bar_t)
                noise = torch.zeros_like(x)

            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)) * predicted_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise

        return x

    @torch.no_grad()
    def sample(self, model, shape, labels=None, guidance_scale=1.0, sampler='ddim', sample_steps=100, initial_noise=None):
        if sampler == 'ddpm':
            return self.p_sample_loop(
                model,
                shape,
                labels=labels,
                guidance_scale=guidance_scale,
                initial_noise=initial_noise,
            )
        return self.ddim_sample_loop(
            model,
            shape,
            labels=labels,
            guidance_scale=guidance_scale,
            sample_steps=sample_steps,
            initial_noise=initial_noise,
        )

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


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(4, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).view(b, c, h * w).permute(0, 2, 1)
        k = self.k(x_norm).view(b, c, h * w)
        v = self.v(x_norm).view(b, c, h * w).permute(0, 2, 1)

        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).contiguous().view(b, c, h, w)
        return x + self.proj(out)


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
    def __init__(self, in_channels=3, model_channels=64, num_classes=10, use_attention=False, num_res_blocks=2):
        super().__init__()
        # Time Embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        
        time_dim = model_channels * 4
        self.num_classes = num_classes
        self.null_class_idx = num_classes
        self.class_embed = nn.Embedding(num_classes + 1, time_dim)
        
        # Initial Conv
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Down 1: 28 -> 14
        self.down1_res = ResnetBlock(model_channels, time_dim)
        self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        # Down 2: 14 -> 7
        self.down2_res = ResnetBlock(model_channels, time_dim, out_dim=model_channels * 2)
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        # Middle
        self.mid_blocks = nn.ModuleList([
            ResnetBlock(model_channels * 2, time_dim)
            for _ in range(num_res_blocks)
        ])
        self.mid_attn = AttentionBlock(model_channels * 2) if use_attention else nn.Identity()
        
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
        
    def forward(self, x, t, labels=None):
        t_emb = self.time_embed(t)
        if labels is None:
            class_ids = torch.full((x.size(0),), self.null_class_idx, device=x.device, dtype=torch.long)
        else:
            class_ids = labels.clone().long()
            class_ids[class_ids < 0] = self.null_class_idx
        t_emb = t_emb + self.class_embed(class_ids)
        
        # Initial
        h_init = self.init_conv(x) # [B, C, 28, 28]
        
        # Down 1
        h1 = self.down1_res(h_init, t_emb) # [B, C, 28, 28]
        h1_pool = self.down1_pool(h1)      # [B, C, 14, 14]
        
        # Down 2
        h2 = self.down2_res(h1_pool, t_emb) # [B, 2C, 14, 14]
        h2_pool = self.down2_pool(h2)       # [B, 2C, 7, 7]
        
        # Middle
        h_mid = h2_pool
        for block in self.mid_blocks:
            h_mid = block(h_mid, t_emb)
        h_mid = self.mid_attn(h_mid)
        
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


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

        # Keep buffers (e.g., norm running stats) synchronized.
        ema_buffers = dict(self.ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, buf in model_buffers.items():
            ema_buffers[name].copy_(buf)


def train_diffusion(
    model,
    loader,
    schedule,
    epochs=20,
    lr=2e-4,
    encode_fn=None,
    ema_decay=0.999,
    cond_dropout_prob=0.1,
    snapshot_epochs=None,
    snapshot_noise=None,
    snapshot_labels=None,
    snapshot_sampler='ddim',
    snapshot_sample_steps=100,
    snapshot_cfg_scale=1.0,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema = EMA(model, decay=ema_decay)
    history = []
    epoch_samples = {}
    snapshot_epoch_set = set(snapshot_epochs or [])
    model.train()

    for epoch in range(epochs):
        epoch_num = epoch + 1
        running = 0.0
        n_batches = 0

        for x, y, _ in tqdm(loader, desc=f'Diff epoch {epoch + 1}/{epochs}', leave=False):
            x = x.to(device)
            y = y.to(device)
            # Keep x in [-1, 1] regardless of loader output range.
            if x.min() >= 0.0:
                x = x * 2.0 - 1.0
            if encode_fn is not None:
                with torch.no_grad():
                    # For Latent Diffusion: encode pixels to latent space
                    mu, _ = encode_fn(x)
                    x = mu

            bs = x.size(0)
            t = torch.randint(0, schedule.num_timesteps, (bs,), device=device).long()
            noise = torch.randn_like(x)
            x_t = schedule.q_sample(x_0=x, t=t, noise=noise)
            cond_labels = y.clone()
            if cond_dropout_prob > 0:
                drop_mask = torch.rand(bs, device=device) < cond_dropout_prob
                cond_labels[drop_mask] = -1
            predicted_noise = model(x_t, t, cond_labels)
            loss = F.mse_loss(predicted_noise, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)

            running += loss.item()
            n_batches += 1

        avg = running / max(n_batches, 1)
        history.append(avg)
        print(f'Diff epoch {epoch_num:02d}/{epochs} | loss: {avg:.4f}')

        if snapshot_noise is not None and snapshot_labels is not None and epoch_num in snapshot_epoch_set:
            with torch.no_grad():
                snap = schedule.sample(
                    ema.ema_model,
                    shape=tuple(snapshot_noise.shape),
                    labels=snapshot_labels,
                    guidance_scale=snapshot_cfg_scale,
                    sampler=snapshot_sampler,
                    sample_steps=snapshot_sample_steps,
                    initial_noise=snapshot_noise,
                )
                snap = torch.clamp((snap + 1.0) / 2.0, 0.0, 1.0)
                epoch_samples[epoch_num] = snap.detach().cpu()

    return history, ema.ema_model, epoch_samples


def plot_mse_loss_curve(history, save_file):
    epochs = np.arange(1, len(history) + 1)
    values = np.asarray(history, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, values, color='tab:blue', linewidth=2.0, label='MSE Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('C-Diffusion Training MSE Over Epochs')
    ax.grid(alpha=0.25)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(save_file, dpi=180)
    plt.close(fig)


def save_generation_progress_grid(epoch_samples, ordered_epochs, save_file):
    num_rows = 3
    num_cols = len(ordered_epochs)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.0 * num_cols, 3.0 * num_rows))

    if num_cols == 1:
        axes = np.asarray(axes).reshape(num_rows, 1)

    for col, ep in enumerate(ordered_epochs):
        batch = epoch_samples[ep]
        for row in range(num_rows):
            ax = axes[row, col]
            img = batch[row].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')
            if row == 0:
                ax.set_title(f'Epoch {ep}')
            if col == 0:
                ax.set_ylabel(f'Image {row + 1}')

    fig.suptitle('C-Diffusion Generation Evolution (fixed seed)')
    fig.tight_layout()
    fig.savefig(save_file, dpi=180)
    plt.close(fig)


def save_cdiffusion_checkpoint(
    checkpoint_path,
    ema_model,
    schedule,
    history,
    num_classes,
    config,
):
    payload = {
        'model_state_dict': ema_model.state_dict(),
        'schedule': {
            'num_timesteps': int(schedule.num_timesteps),
        },
        'history_mse': [float(v) for v in history],
        'num_classes': int(num_classes),
        'config': config,
    }
    torch.save(payload, checkpoint_path)


@torch.no_grad()
def generate_conditional_diffusion_samples(
    model,
    schedule,
    count,
    device,
    num_classes=10,
    cfg_scale=1.0,
    sampler='ddim',
    sample_steps=100,
):
    model.eval()
    samples = []
    batch_size = 64
    generated = 0
    while generated < count:
        cur_bs = min(batch_size, count - generated)
        labels = torch.randint(0, num_classes, (cur_bs,), device=device)
        batch = schedule.sample(
            model,
            shape=(cur_bs, 3, 32, 32),
            labels=labels,
            guidance_scale=cfg_scale,
            sampler=sampler,
            sample_steps=sample_steps,
        )
        batch = torch.clamp((batch + 1.0) / 2.0, 0.0, 1.0)
        samples.append(batch.cpu())
        generated += cur_bs
    return torch.cat(samples, dim=0)


def evaluate_conditional_diffusion_protocol(
    model,
    schedule,
    loader,
    device,
    feature_extractor,
    num_classes=10,
    cfg_scale=1.0,
    sampler='ddim',
    sample_steps=100,
):
    fid_scores = []
    kid_means = []

    real_images = get_real_samples(loader, count=EVAL_CONFIG['reference_count'])
    real_features = extract_inception_features(real_images, feature_extractor, device=device)
    mu_real, sigma_real = feature_statistics(real_features)

    for seed in range(EVAL_CONFIG['num_seeds']):
        torch.manual_seed(seed)
        print(f"  Evaluating seed {seed+1}/{EVAL_CONFIG['num_seeds']}...")

        gen_images = generate_conditional_diffusion_samples(
            model=model,
            schedule=schedule,
            count=EVAL_CONFIG['generated_count'],
            device=device,
            num_classes=num_classes,
            cfg_scale=cfg_scale,
            sampler=sampler,
            sample_steps=sample_steps,
        )
        gen_features = extract_inception_features(gen_images, feature_extractor, device=device)
        mu_gen, sigma_gen = feature_statistics(gen_features)
        fid = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        fid_scores.append(fid)

        k_mean, _ = kid_score(
            real_features,
            gen_features,
            subset_size=EVAL_CONFIG['kid_subset_size'],
            num_subsets=EVAL_CONFIG['kid_subsets'],
        )
        kid_means.append(k_mean)

    print(f"\nResults for C-DIFFUSION over {EVAL_CONFIG['num_seeds']} seeds:")
    print(f"  FID: {np.mean(fid_scores):.4f} ± {np.std(fid_scores):.4f}")
    print(f"  KID: {np.mean(kid_means):.4f} ± {np.std(kid_means):.4f}\n")


def evaluate_conditional_diffusion_metrics(
    model,
    schedule,
    loader,
    device,
    feature_extractor,
    num_classes=10,
    eval_count=1000,
    num_seeds=3,
    cfg_scale=1.0,
    sampler='ddim',
    sample_steps=100,
):
    fid_scores = []
    kid_scores = []

    real_images = get_real_samples(loader, count=eval_count)
    real_features = extract_inception_features(real_images, feature_extractor, device=device)
    mu_real, sigma_real = feature_statistics(real_features)

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        gen_images = generate_conditional_diffusion_samples(
            model=model,
            schedule=schedule,
            count=eval_count,
            device=device,
            num_classes=num_classes,
            cfg_scale=cfg_scale,
            sampler=sampler,
            sample_steps=sample_steps,
        )
        gen_features = extract_inception_features(gen_images, feature_extractor, device=device)
        mu_gen, sigma_gen = feature_statistics(gen_features)
        fid = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        kid_mean, _ = kid_score(
            real_features,
            gen_features,
            subset_size=min(100, eval_count // 2),
            num_subsets=20,
            seed=seed + 123,
        )
        fid_scores.append(fid)
        kid_scores.append(kid_mean)

    return {
        'fid_mean': float(np.mean(fid_scores)),
        'fid_std': float(np.std(fid_scores)),
        'kid_mean': float(np.mean(kid_scores)),
        'kid_std': float(np.std(kid_scores)),
    }


def run_cdiffusion_pipeline(
    train_loader,
    test_loader,
    num_classes,
    learning_rate,
    base_channels,
    use_attention,
    num_res_blocks,
    timesteps,
    epochs,
    beta_schedule,
    ema_decay,
    cond_dropout_prob,
    cfg_scale,
    sampler,
    sample_steps,
    feature_extractor,
    eval_count,
    eval_seeds,
    save_path,
    run_name,
    save_samples=True,
    snapshot_epochs=None,
    snapshot_noise=None,
    snapshot_labels=None,
):
    schedule = GaussianDiffusion(num_timesteps=timesteps, device=device, beta_schedule=beta_schedule)
    diff_model = PixelUNet(
        in_channels=3,
        model_channels=base_channels,
        num_classes=num_classes,
        use_attention=use_attention,
        num_res_blocks=num_res_blocks,
    ).to(device)

    print(
        f"[{run_name}] Starting C-Diffusion training "
        f"(lr={learning_rate:.2e}, base_channels={base_channels}, "
        f"use_attention={use_attention}, num_res_blocks={num_res_blocks}, "
        f"timesteps={timesteps}, epochs={epochs}, sampler={sampler}, sample_steps={sample_steps})"
    )
    history, ema_model, epoch_samples = train_diffusion(
        model=diff_model,
        loader=train_loader,
        schedule=schedule,
        epochs=epochs,
        lr=learning_rate,
        ema_decay=ema_decay,
        cond_dropout_prob=cond_dropout_prob,
        snapshot_epochs=snapshot_epochs,
        snapshot_noise=snapshot_noise,
        snapshot_labels=snapshot_labels,
        snapshot_sampler=sampler,
        snapshot_sample_steps=sample_steps,
        snapshot_cfg_scale=cfg_scale,
    )

    if save_samples:
        with torch.no_grad():
            class_grid = torch.arange(min(10, num_classes), device=device).repeat_interleave(10)
            samples = schedule.sample(
                ema_model,
                shape=(class_grid.size(0), 3, 32, 32),
                labels=class_grid,
                guidance_scale=cfg_scale,
                sampler=sampler,
                sample_steps=sample_steps,
            )
            samples = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)
            save_image(samples, save_path / f'{run_name}_generated_samples.png', nrow=10)

    metrics = evaluate_conditional_diffusion_metrics(
        model=ema_model,
        schedule=schedule,
        loader=test_loader,
        device=device,
        feature_extractor=feature_extractor,
        num_classes=num_classes,
        eval_count=eval_count,
        num_seeds=eval_seeds,
        cfg_scale=cfg_scale,
        sampler=sampler,
        sample_steps=sample_steps,
    )
    print(
        f"[{run_name}] FID: {metrics['fid_mean']:.4f} +/- {metrics['fid_std']:.4f} | "
        f"KID: {metrics['kid_mean']:.4f} +/- {metrics['kid_std']:.4f}"
    )

    return {
        'ema_model': ema_model,
        'schedule': schedule,
        'history': history,
        'epoch_samples': epoch_samples,
        **metrics,
    }


def main():
    # Device
    dev = device
    print(f"Using device: {dev}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-20pct', action='store_true', help='Use training_20_percent.csv subset')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per run/trial')
    parser.add_argument('--beta-schedule', choices=['linear', 'cosine'], default='cosine', help='Diffusion beta schedule')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay for diffusion model weights')
    parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddim', help='Sampler for generation/evaluation')
    parser.add_argument('--sample-steps', type=int, default=100, help='Sampling steps for DDIM (recommended 50-100)')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate for non-search mode')
    parser.add_argument('--base-channels', type=int, default=64, choices=[32, 64, 96], help='Base channels for UNet')
    parser.add_argument('--use-attention', action='store_true', help='Enable bottleneck attention in UNet')
    parser.add_argument('--num-res-blocks', type=int, default=2, choices=[2, 3], help='Number of bottleneck residual blocks')
    parser.add_argument('--bayes-search', action='store_true', help='Run Bayesian search with Optuna (TPE sampler)')
    parser.add_argument('--n-trials', type=int, default=10, help='Number of Bayesian search trials')
    parser.add_argument('--study-name', type=str, default='cdiffusion_bayes_search', help='Optuna study name used for persistence/resume')
    parser.add_argument('--study-db', type=str, default=None, help='Optional path to Optuna SQLite DB (default: src/results/cdiffusion_bayes_search.db)')
    parser.add_argument('--skip-eval', action='store_true', help='Skip FID/KID evaluation phase')
    parser.add_argument('--eval-count', type=int, default=None, help='Alias: sets both --eval-reference-count and --eval-generated-count')
    parser.add_argument('--eval-reference-count', type=int, default=5000, help='Number of real reference images for evaluation')
    parser.add_argument('--eval-generated-count', type=int, default=5000, help='Number of generated images for evaluation')
    parser.add_argument('--eval-seeds', type=int, default=10, help='Number of repeated seeds for protocol evaluation')
    parser.add_argument('--cond-drop-prob', type=float, default=0.1, help='Condition dropout probability for CFG training')
    parser.add_argument('--cfg-scale', type=float, default=6.0, help='Classifier-free guidance scale for sampling/evaluation')
    parser.add_argument('--progress-seed', type=int, default=42, help='Seed used for fixed progression samples')
    parser.add_argument('--save-checkpoint', dest='save_checkpoint', action='store_true', help='Save final EMA checkpoint for downstream analysis')
    parser.add_argument('--no-save-checkpoint', dest='save_checkpoint', action='store_false', help='Disable final checkpoint export')
    parser.add_argument('--checkpoint-name', type=str, default='cdiffusion_final_checkpoint.pt', help='Filename for final checkpoint in src/results')
    parser.set_defaults(save_checkpoint=True)
    args = parser.parse_args()
    env_flag = os.environ.get('USE_20_PERCENT', '')
    use_subset = args.use_20pct or (env_flag.lower() in ('1', 'true', 'yes'))

    # Alias support to keep CLI naming consistent with GAN scripts.
    if args.eval_count is not None:
        args.eval_reference_count = args.eval_count
        args.eval_generated_count = args.eval_count

    if use_subset:
        print('Loading 20% subset via training_20_percent.csv')
        cfg = setup_artbench_from_csv_subset(
            project_root=None,
            training_csv_path=None,
            image_size=32,
            batch_size=DEFAULT_BATCH_SIZE,
            num_workers=DEFAULT_NUM_WORKERS,
            shuffle=True,
        )
        root = cfg['project_root']
        train_hf = cfg['train_hf']
        class_names = cfg['class_names']
        train_loader = cfg['train_loader']
    else:
        # Load full ArtBench training split
        root = Path(__file__).resolve().parent.parent
        train_hf, class_names = load_artbench_train_split(root)
        transform = build_transform(image_size=32)
        train_ds = HFDatasetTorch(train_hf, transform=transform)
        train_loader = DataLoader(
            train_ds,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    # Always evaluate against the test split.
    _, test_hf, _ = load_artbench_splits(root)
    test_transform = build_transform(image_size=32)
    test_ds = HFDatasetTorch(test_hf, transform=test_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = len(class_names)

    # Lock search-space constants requested by the project spec.
    DIFF_TIMESTEPS = 1000
    DIFF_EPOCHS = args.epochs

    save_path = Path(__file__).resolve().parent / 'results'
    save_path.mkdir(exist_ok=True)

    feat_extractor = build_feature_extractor(dev)

    if args.bayes_search:
        if optuna is None:
            raise ImportError('Optuna is required for --bayes-search. Install with: pip install optuna')

        print('Starting Bayesian hyperparameter search for C-Diffusion...')
        trial_dir = save_path / 'bayes_trials_cdiffusion'
        trial_dir.mkdir(exist_ok=True)

        study_db = Path(args.study_db) if args.study_db else (save_path / 'cdiffusion_bayes_search.db')
        study_db.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{study_db.resolve()}"

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name=args.study_name,
            storage=storage_url,
            load_if_exists=True,
        )

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        remaining_trials = max(0, args.n_trials - len(completed_trials))
        print(
            f"Using study='{args.study_name}' at {study_db} | "
            f"completed={len(completed_trials)}, target={args.n_trials}, remaining={remaining_trials}"
        )

        def objective(trial):
            learning_rate = trial.suggest_float('learning_rate', 2e-5, 5e-4, log=True)
            base_channels = trial.suggest_categorical('base_channels', [32, 64, 96])
            use_attention = trial.suggest_categorical('use_attention', [True, False])
            num_res_blocks = trial.suggest_categorical('num_res_blocks', [2, 3])

            trial_name = f'trial_{trial.number:03d}'
            trial_result = run_cdiffusion_pipeline(
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=num_classes,
                learning_rate=learning_rate,
                base_channels=base_channels,
                use_attention=use_attention,
                num_res_blocks=num_res_blocks,
                timesteps=DIFF_TIMESTEPS,
                epochs=DIFF_EPOCHS,
                beta_schedule=args.beta_schedule,
                ema_decay=args.ema_decay,
                cond_dropout_prob=args.cond_drop_prob,
                cfg_scale=args.cfg_scale,
                sampler=args.sampler,
                sample_steps=args.sample_steps,
                feature_extractor=feat_extractor,
                eval_count=args.eval_generated_count,
                eval_seeds=args.eval_seeds,
                save_path=trial_dir,
                run_name=trial_name,
                save_samples=True,
            )

            trial.set_user_attr('kid_mean', trial_result['kid_mean'])
            trial.set_user_attr('kid_std', trial_result['kid_std'])
            return trial_result['fid_mean']

        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials)
        else:
            print('Target number of completed trials already reached; skipping optimization.')

        best = study.best_trial
        best_params = best.params
        print(f"Best trial #{best.number} -> FID: {best.value:.4f}")
        print(f"Best params: {best_params}")

        best_params_path = save_path / 'cdiffusion_bayes_best_params.json'
        with best_params_path.open('w', encoding='utf-8') as f:
            json.dump(
                {
                    'best_trial': best.number,
                    'best_fid': float(best.value),
                    'params': best_params,
                    'kid_mean': best.user_attrs.get('kid_mean'),
                    'kid_std': best.user_attrs.get('kid_std'),
                },
                f,
                indent=2,
            )
        print(f'Saved best search result to {best_params_path}')

        best_result = run_cdiffusion_pipeline(
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            learning_rate=float(best_params['learning_rate']),
            base_channels=int(best_params['base_channels']),
            use_attention=bool(best_params['use_attention']),
            num_res_blocks=int(best_params['num_res_blocks']),
            timesteps=DIFF_TIMESTEPS,
            epochs=DIFF_EPOCHS,
            beta_schedule=args.beta_schedule,
            ema_decay=args.ema_decay,
            cond_dropout_prob=args.cond_drop_prob,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            sample_steps=args.sample_steps,
            feature_extractor=feat_extractor,
            eval_count=args.eval_generated_count,
            eval_seeds=args.eval_seeds,
            save_path=save_path,
            run_name='cdiffusion_best_bayes',
            save_samples=True,
        )

        ema_model = best_result['ema_model']
        schedule = best_result['schedule']
        diff_history = best_result['history']
        epoch_samples = best_result.get('epoch_samples', {})
    else:
        requested_milestones = [1, 50, 100, 200]
        snapshot_epochs = [ep for ep in requested_milestones if ep <= DIFF_EPOCHS]
        progress_generator = torch.Generator(device=dev)
        progress_generator.manual_seed(args.progress_seed)
        snapshot_noise = torch.randn(3, 3, 32, 32, generator=progress_generator, device=dev)
        snapshot_labels = torch.randint(0, num_classes, (3,), generator=progress_generator, device=dev)

        default_result = run_cdiffusion_pipeline(
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            learning_rate=args.learning_rate,
            base_channels=args.base_channels,
            use_attention=args.use_attention,
            num_res_blocks=args.num_res_blocks,
            timesteps=DIFF_TIMESTEPS,
            epochs=DIFF_EPOCHS,
            beta_schedule=args.beta_schedule,
            ema_decay=args.ema_decay,
            cond_dropout_prob=args.cond_drop_prob,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            sample_steps=args.sample_steps,
            feature_extractor=feat_extractor,
            eval_count=args.eval_generated_count,
            eval_seeds=args.eval_seeds,
            save_path=save_path,
            run_name='cdiffusion_generated_samples',
            save_samples=True,
            snapshot_epochs=snapshot_epochs,
            snapshot_noise=snapshot_noise,
            snapshot_labels=snapshot_labels,
        )
        ema_model = default_result['ema_model']
        schedule = default_result['schedule']
        diff_history = default_result['history']
        epoch_samples = default_result.get('epoch_samples', {})

        if (not use_subset) and (DIFF_EPOCHS >= 200):
            mse_plot_path = save_path / 'cdiffusion_mse_curve_200ep.png'
            plot_mse_loss_curve(diff_history, mse_plot_path)
            print(f'Saved MSE curve to {mse_plot_path}')

            if all(ep in epoch_samples for ep in requested_milestones):
                progress_path = save_path / f'cdiffusion_generation_progress_seed_{args.progress_seed}.png'
                save_generation_progress_grid(epoch_samples, requested_milestones, progress_path)
                print(f'Saved epoch progression grid to {progress_path}')
            else:
                print('Progression grid skipped: missing one or more milestone snapshots (1, 50, 100, 200).')

        if args.save_checkpoint:
            checkpoint_path = save_path / args.checkpoint_name
            checkpoint_cfg = {
                'learning_rate': float(args.learning_rate),
                'base_channels': int(args.base_channels),
                'use_attention': bool(args.use_attention),
                'num_res_blocks': int(args.num_res_blocks),
                'timesteps': int(DIFF_TIMESTEPS),
                'epochs': int(DIFF_EPOCHS),
                'beta_schedule': args.beta_schedule,
                'ema_decay': float(args.ema_decay),
                'cond_drop_prob': float(args.cond_drop_prob),
                'cfg_scale': float(args.cfg_scale),
                'sampler': args.sampler,
                'sample_steps': int(args.sample_steps),
                'eval_count': int(args.eval_generated_count),
                'eval_seeds': int(args.eval_seeds),
                'use_20pct': bool(use_subset),
                'progress_seed': int(args.progress_seed),
            }
            save_cdiffusion_checkpoint(
                checkpoint_path=checkpoint_path,
                ema_model=ema_model,
                schedule=schedule,
                history=diff_history,
                num_classes=num_classes,
                config=checkpoint_cfg,
            )
            print(f'Saved C-Diffusion checkpoint to {checkpoint_path}')

    # --- Evaluation: FID & KID ---
    if args.skip_eval:
        print('Skipping evaluation phase (--skip-eval).')
    else:
        EVAL_CONFIG['reference_count'] = args.eval_reference_count
        EVAL_CONFIG['generated_count'] = args.eval_generated_count
        EVAL_CONFIG['num_seeds'] = args.eval_seeds
        print(
            f"Evaluation config: refs={EVAL_CONFIG['reference_count']}, "
            f"gen={EVAL_CONFIG['generated_count']}, seeds={EVAL_CONFIG['num_seeds']}"
        )
        base_evaluation(None, 'baseline', test_loader, dev, feature_extractor=feat_extractor)
        evaluate_conditional_diffusion_protocol(
            model=ema_model,
            schedule=schedule,
            loader=test_loader,
            device=dev,
            feature_extractor=feat_extractor,
            num_classes=num_classes,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            sample_steps=args.sample_steps,
        )

    return ema_model, diff_history


if __name__ == '__main__':
    main()