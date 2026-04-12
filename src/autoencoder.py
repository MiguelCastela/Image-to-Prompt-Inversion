import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32, num_classes=10, class_embed_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        self.class_embed = nn.Embedding(num_classes, class_embed_dim)

        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # If we spatially concatenate the class embedding as extra channels,
        # the flattened encoder input dimension becomes (128 + class_embed_dim) * 8 * 8
        enc_in_dim = (128 + class_embed_dim) * 8 * 8
        self.fc_mu = nn.Linear(enc_in_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_in_dim, latent_dim)

        dec_in_dim = latent_dim + class_embed_dim
        self.dec_fc = nn.Linear(dec_in_dim, 128 * 8 * 8)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x, y):
        h = self.enc_conv(x)
        # Spatially expand the class embedding and concatenate as extra channels
        y_embed_spatial = self.class_embed(y).view(h.size(0), self.class_embed_dim, 1, 1)
        y_embed_spatial = y_embed_spatial.expand(-1, -1, h.size(2), h.size(3))
        h = torch.cat([h, y_embed_spatial], dim=1)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, y):
        y_embed = self.class_embed(y)
        z_cond = torch.cat([z, y_embed], dim=1)
        h = self.dec_fc(z_cond).view(-1, 128, 8, 8)
        return self.dec_conv(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z, y)
        return xhat, mu, logvar


def vae_loss(xhat, x, mu, logvar, beta=0.7):
    # Reconstruction term: sum over pixels, then convert to per-sample value
    b = x.size(0)
    recon_sum = F.binary_cross_entropy(xhat, x, reduction='sum')
    # KL divergence (sum over latent dims), per-batch sum
    kl_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Convert to per-sample values for stable bookkeeping
    recon = recon_sum / b
    kl = kl_sum / b
    loss = recon + beta * kl
    return loss, recon, kl


def train_vae(model, loader, optimizer, epochs=20, beta=0.7):
    model.train()
    hist = []
    for ep in range(epochs):
        tl, tr, tk = 0.0, 0.0, 0.0
        for x, y, _ in tqdm(loader, leave=False):
            x = x.to(device)
            y = y.to(device)
            xhat, mu, logvar = model(x, y)

            b = x.size(0)
            # compute sums then convert to per-sample inside vae_loss
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item() * b
            tr += recon.item() * b
            tk += kl.item() * b
        n = len(loader.dataset)
        hist.append({'train_loss': tl / n, 'train_recon_bce': tr / n, 'train_kl': tk / n})
        print(f'Epoch {ep+1}/{epochs} | train_loss={tl/n:.4f} train_recon={tr/n:.4f} train_kl={tk/n:.4f}')
    return hist


def evaluate_vae(model, loader, beta=0.7):
    model.eval()
    tl, tr, tk, tm, ta, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            xhat, mu, logvar = model(x, y)
            b = x.size(0)

            # use sums internally and convert to per-sample in vae_loss
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)

            # accumulate reconstruction / kl totals and pixel metrics
            tl += loss.item() * b
            tr += recon.item() * b
            tk += kl.item() * b
            tm += F.mse_loss(xhat, x, reduction='sum').item()
            ta += F.l1_loss(xhat, x, reduction='sum').item()
            n += b
    numel = x[0].numel()
    return {'loss': tl / n, 'recon_bce': tr / n, 'kl': tk / n, 'mse': tm / (n * numel), 'mae': ta / (n * numel)}