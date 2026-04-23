import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

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
from evaluation import EVAL_CONFIG, build_feature_extractor, base_evaluation, evaluate_model_protocol

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


def plot_recon_kl_curves(history, save_file):
    epochs = [i + 1 for i in range(len(history))]
    recon = [h['train_recon_bce'] for h in history]
    kl = [h['train_kl'] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(epochs, recon, color='tab:blue', linewidth=2.0, label='Reconstruction Loss')
    l2 = ax2.plot(epochs, kl, color='tab:red', linewidth=2.0, label='KL Divergence')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss (BCE)', color='tab:blue')
    ax2.set_ylabel('KL Divergence', color='tab:red')
    ax1.tick_params(axis='y', colors='tab:blue')
    ax2.tick_params(axis='y', colors='tab:red')
    ax1.grid(alpha=0.25)

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best')
    fig.tight_layout()
    fig.savefig(save_file, dpi=180)
    plt.close(fig)


def save_generation_progress_grid(epoch_samples, ordered_epochs, save_file):
    num_rows = 3
    num_cols = len(ordered_epochs)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.0 * num_cols, 3.0 * num_rows))

    if num_cols == 1:
        axes = axes.reshape(num_rows, 1)

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

    fig.suptitle('Generation Evolution (fixed seed)')
    fig.tight_layout()
    fig.savefig(save_file, dpi=180)
    plt.close(fig)


def train_vae(
    model,
    loader,
    optimizer,
    epochs=20,
    beta=0.7,
    warmup_epochs=5,
    snapshot_epochs=None,
    snapshot_latents=None,
):
    model.train()
    hist = []
    epoch_samples = {}
    snapshot_epoch_set = set(snapshot_epochs or [])

    for ep in range(epochs):
        epoch_num = ep + 1
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
        print(f'Epoch {epoch_num}/{epochs} | beta={current_beta:.4f} train_loss={tl/n:.4f} train_recon={tr/n:.4f} train_kl={tk/n:.4f}')

        if snapshot_latents is not None and epoch_num in snapshot_epoch_set:
            model.eval()
            with torch.no_grad():
                snap = model.decode(snapshot_latents).detach().cpu().clamp(0.0, 1.0)
            epoch_samples[epoch_num] = snap
            model.train()

    return hist, epoch_samples


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
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--eval-count', type=int, default=5000, help='Number of images for real/generated evaluation')
    parser.add_argument('--eval-seeds', type=int, default=10, help='Number of evaluation seeds')
    parser.add_argument('--progress-seed', type=int, default=42, help='Seed for fixed latent vectors used in progression grid')
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
    EPOCHS = args.epochs
    LR = 1e-3
    BETA = 0.5

    EVAL_CONFIG['reference_count'] = args.eval_count
    EVAL_CONFIG['generated_count'] = args.eval_count
    EVAL_CONFIG['num_seeds'] = args.eval_seeds

    model = ConvVAE(latent_dim=LATENT_DIM).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    save_path = Path(__file__).resolve().parent / 'results'
    save_path.mkdir(exist_ok=True)

    requested_milestones = [1, 50, 100, 200]
    snapshot_epochs = [ep for ep in requested_milestones if ep <= EPOCHS]
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.progress_seed)
    snapshot_latents = torch.randn(3, LATENT_DIM, device=dev, generator=gen)

    print("Starting VAE training (non-conditional) on full ArtBench training split...")
    history, epoch_samples = train_vae(
        model,
        train_loader,
        optimizer,
        epochs=EPOCHS,
        beta=BETA,
        snapshot_epochs=snapshot_epochs,
        snapshot_latents=snapshot_latents,
    )

    print("Training complete.")

    should_export_final_artifacts = (not use_subset) and (EPOCHS >= 200)
    if should_export_final_artifacts:
        loss_plot_path = save_path / 'vae_recon_kl_curves_200ep.png'
        plot_recon_kl_curves(history, loss_plot_path)
        print(f"Saved reconstruction/KL curve to {loss_plot_path}")

        if all(ep in epoch_samples for ep in requested_milestones):
            progression_path = save_path / f'vae_generation_progress_seed_{args.progress_seed}.png'
            save_generation_progress_grid(epoch_samples, requested_milestones, progression_path)
            print(f"Saved epoch progression grid to {progression_path}")
        else:
            print('Progression grid skipped: missing one or more milestone snapshots (1, 50, 100, 200).')

    # Generate and save 100 samples for non-conditional VAE.
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
