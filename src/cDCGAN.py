import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from torchvision.utils import make_grid

# Setup constants for ArtBench-10
NUM_CLASSES = 10
LATENT_DIM = 100
IMG_CHANNELS = 3

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

device = get_device()

# --- MODELS ---

class CGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, image_channels=3, ngf=64):
        super().__init__()
        # Label embedding: maps class index to a vector of size latent_dim
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        self.net = nn.Sequential(
            # Input: latent_dim + label_emb_dim = 200
            nn.ConvTranspose2d(latent_dim * 2, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z, labels):
        # Concatenate noise and label embedding
        le = self.label_emb(labels).view(z.size(0), z.size(1), 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = torch.cat([z, le], dim=1)
        return self.net(x)

class CDiscriminator(nn.Module):
    def __init__(self, image_channels=3, num_classes=10, ndf=64):
        super().__init__()
        # Label embedding maps to a 1x32x32 feature map to append as a channel
        self.label_emb = nn.Embedding(num_classes, 32 * 32)
        
        self.net = nn.Sequential(
            # Input: 3 image channels + 1 label channel = 4
            nn.Conv2d(image_channels + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        # Reshape label embedding to match image dimensions
        le = self.label_emb(labels).view(-1, 1, 32, 32)
        x = torch.cat([x, le], dim=1)
        return self.net(x).view(-1, 1)

def init_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- TRAINING ---

def train_cgan(generator, discriminator, loader, latent_dim, epochs=20, lr=2e-4):
    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {'g_loss': [], 'd_loss': []}
    
    for epoch in range(epochs):
        g_running, d_running, n_batches = 0.0, 0.0, 0
        for real_imgs, labels, _ in tqdm(loader, desc=f'Epoch {epoch+1}'):
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            bs = real_imgs.size(0)

            # Labels for BCE
            real_target = torch.ones(bs, 1, device=device)
            fake_target = torch.zeros(bs, 1, device=device)

            # --- Discriminator update ---
            opt_d.zero_grad()
            # Real pass
            d_real = discriminator(real_imgs, labels)
            loss_real = criterion(d_real, real_target)
            # Fake pass
            z = torch.randn(bs, latent_dim, device=device)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            fake_imgs = generator(z, gen_labels)
            d_fake = discriminator(fake_imgs.detach(), gen_labels)
            loss_fake = criterion(d_fake, fake_target)
            
            d_loss = loss_real + loss_fake
            d_loss.backward()
            opt_d.step()

            # --- Generator update ---
            opt_g.zero_grad()
            # Generator wants Discriminator to think fake is real
            d_g_fake = discriminator(fake_imgs, gen_labels)
            g_loss = criterion(d_g_fake, real_target)
            g_loss.backward()
            opt_g.step()

            # Bookkeeping
            g_running += g_loss.item()
            d_running += d_loss.item()
            n_batches += 1

        history['g_loss'].append(g_running / n_batches)
        history['d_loss'].append(d_running / n_batches)
        print(f"Epoch {epoch+1} | D Loss: {history['d_loss'][-1]:.4f} | G Loss: {history['g_loss'][-1]:.4f}")

    return history

# --- INFERENCE & VISUALIZATION ---

@torch.no_grad()
def run_inference(generator, latent_dim, n_samples=10, specific_class=None):
    generator.eval()
    z = torch.randn(n_samples, latent_dim, device=device)
    if specific_class is not None:
        labels = torch.full((n_samples,), specific_class, dtype=torch.long, device=device)
    else:
        labels = torch.randint(0, NUM_CLASSES, (n_samples,), device=device)
    
    samples = generator(z, labels)
    # Already in [0, 1] from Sigmoid
    return samples

@torch.no_grad()
def latent_walk(generator, latent_dim, label=0, steps=10):
    generator.eval()
    z0 = torch.randn(1, latent_dim, device=device)
    z1 = torch.randn(1, latent_dim, device=device)
    
    alphas = torch.linspace(0, 1, steps, device=device)
    z = torch.cat([(1 - a) * z0 + a * z1 for a in alphas])
    labels = torch.full((steps,), label, dtype=torch.long, device=device)
    
    samples = generator(z, labels)
    return samples