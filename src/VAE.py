import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image

# data loading utilities
import argparse
import os

from data_loader import (
    load_artbench_train_split,
    load_artbench_splits,
    build_transform,
    HFDatasetTorch,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    setup_artbench_from_csv_subset,
)
from evaluation import build_feature_extractor, base_evaluation, evaluate_model_protocol

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

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

        enc_in_dim = 128 * 8 * 8
        self.fc_mu = nn.Linear(enc_in_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_in_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z, y=None):
        # Accept an optional label argument for compatibility with
        # evaluation.generate_samples which calls decode(z, labels).
        h = self.dec_fc(z).view(-1, 128, 8, 8)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar


def vae_loss(xhat, x, mu, logvar, beta=0.7):
    b = x.size(0)
    recon_sum = F.binary_cross_entropy(xhat, x, reduction='sum')
    kl_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon = recon_sum / b
    kl = kl_sum / b
    loss = recon + beta * kl
    return loss, recon, kl


def _to_unit_interval(x):
    # BCE with sigmoid decoder expects targets in [0, 1].
    if x.min() < 0.0:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def train_vae(model, loader, optimizer, epochs=20, beta=0.7, warmup_epochs=5):
    model.train()
    hist = []
    for ep in range(epochs):
        tl, tr, tk = 0.0, 0.0, 0.0
        current_beta = beta * min(1.0, ep / max(warmup_epochs, 1))
        for x, _, _ in tqdm(loader, leave=False):
            x = x.to(device)
            x = _to_unit_interval(x)
            xhat, mu, logvar = model(x)

            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=current_beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            b = x.size(0)
            tl += loss.item() * b
            tr += recon.item() * b
            tk += kl.item() * b
        n = len(loader.dataset)
        hist.append({'train_loss': tl / n, 'train_recon_bce': tr / n, 'train_kl': tk / n, 'beta': current_beta})
        print(f'Epoch {ep+1}/{epochs} | beta={current_beta:.4f} train_loss={tl/n:.4f} train_recon={tr/n:.4f} train_kl={tk/n:.4f}')
    return hist


def evaluate_vae(model, loader, beta=0.7):
    model.eval()
    tl, tr, tk, tm, ta, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            x = _to_unit_interval(x)
            xhat, mu, logvar = model(x)
            b = x.size(0)
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)
            tl += loss.item() * b
            tr += recon.item() * b
            tk += kl.item() * b
            tm += F.mse_loss(xhat, x, reduction='sum').item()
            ta += F.l1_loss(xhat, x, reduction='sum').item()
            n += b
    numel = x[0].numel()
    return {'loss': tl / n, 'recon_bce': tr / n, 'kl': tk / n, 'mse': tm / (n * numel), 'mae': ta / (n * numel)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-20pct', action='store_true', help='Use training_20_percent.csv subset')
    args = parser.parse_args()

    # allow environment variable override
    env_flag = os.environ.get('USE_20_PERCENT', '')
    use_subset = args.use_20pct or (env_flag.lower() in ('1', 'true', 'yes'))

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")

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
        print('Loading full ArtBench training split')
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

    LATENT_DIM = 64
    EPOCHS = 50
    LR = 1e-3
    BETA = 0.5

    model = ConvVAE(latent_dim=LATENT_DIM).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting VAE training (non-conditional) on full ArtBench training split...")
    history = train_vae(model, train_loader, optimizer, epochs=EPOCHS, beta=BETA)

    print("Training complete.")

    # Generate and save 100 samples for non-conditional VAE.
    save_path = Path(__file__).resolve().parent / 'results'
    save_path.mkdir(exist_ok=True)
    with torch.no_grad():
        model.eval()
        z = torch.randn(100, LATENT_DIM, device=dev)
        vae_samples = model.decode(z)
        save_image(vae_samples, save_path / 'vae_nc_generated_samples.png', nrow=10)
        print(f"Saved VAE samples to {save_path}")

    feat_extractor = build_feature_extractor(dev)
    base_evaluation(None, 'baseline', test_loader, dev, feature_extractor=feat_extractor, latent_dim=LATENT_DIM)
    model.eval()
    evaluate_model_protocol(model, 'vae', test_loader, dev, feat_extractor, latent_dim=LATENT_DIM)

    return model, history


if __name__ == "__main__":
    main()
