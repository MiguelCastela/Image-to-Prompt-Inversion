import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from data_loader import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    HFDatasetTorch,
    build_transform,
    load_artbench_splits,
    load_artbench_train_split,
    setup_artbench_from_csv_subset,
)
from evaluation import (
    base_evaluation,
    build_feature_extractor,
    extract_inception_features,
    feature_statistics,
    frechet_distance,
    get_real_samples,
    kid_score,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs):
        batch_size, channels, height, width = inputs.shape
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, channels)

        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_inputs.dtype)
        quantized = encodings @ self.embedding.weight
        quantized = quantized.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class ConvVQVAE(nn.Module):
    def __init__(self, embedding_dim=64, num_embeddings=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_hw = 4

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embedding_dim, 1),
        )

        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, y=None):
        if z.dim() == 2:
            z = z.view(z.size(0), self.embedding_dim, 1, 1).expand(-1, -1, self.latent_hw, self.latent_hw)
        return self.decoder(z)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, vq_loss, perplexity = self.quantizer(z_e)
        xhat = self.decode(z_q)
        return xhat, vq_loss, perplexity

    @torch.no_grad()
    def sample(self, n, device):
        indices = torch.randint(
            0,
            self.quantizer.num_embeddings,
            (n, self.latent_hw, self.latent_hw),
            device=device,
        )
        z_q = self.quantizer.embedding(indices).permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q)


def _to_unit_interval(x):
    if x.min() < 0.0:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def train_vqvae(model, loader, optimizer, epochs=20):
    model.train()
    history = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        total_perplexity = 0.0
        total_samples = 0

        for x, _, _ in tqdm(loader, desc=f'VQ-VAE epoch {epoch + 1}/{epochs}', leave=False):
            x = x.to(device)
            x = _to_unit_interval(x)

            xhat, vq_loss, perplexity = model(x)
            recon_loss = F.binary_cross_entropy(xhat, x, reduction='mean')
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            b = x.size(0)
            total_loss += loss.item() * b
            total_recon += recon_loss.item() * b
            total_vq += vq_loss.item() * b
            total_perplexity += perplexity.item() * b
            total_samples += b

        history.append(
            {
                'train_loss': total_loss / total_samples,
                'train_recon_bce': total_recon / total_samples,
                'train_vq_loss': total_vq / total_samples,
                'train_perplexity': total_perplexity / total_samples,
            }
        )
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"loss={history[-1]['train_loss']:.4f} | "
            f"recon={history[-1]['train_recon_bce']:.4f} | "
            f"vq={history[-1]['train_vq_loss']:.4f} | "
            f"perplexity={history[-1]['train_perplexity']:.2f}"
        )

    return history


@torch.no_grad()
def evaluate_vqvae_metrics(model, loader, feature_extractor, eval_count=1000, num_seeds=3):
    fid_scores = []
    kid_scores = []

    real_images = get_real_samples(loader, count=eval_count)
    real_features = extract_inception_features(real_images, feature_extractor, device=device)
    mu_real, sigma_real = feature_statistics(real_features)

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        generated = []
        total = 0
        while total < eval_count:
            cur_bs = min(64, eval_count - total)
            batch = model.sample(cur_bs, device=device)
            generated.append(batch.cpu())
            total += cur_bs

        gen_images = torch.cat(generated, dim=0)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-20pct', action='store_true', help='Use training_20_percent.csv subset')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--embedding-dim', type=int, default=64, help='VQ embedding dimension')
    parser.add_argument('--num-embeddings', type=int, default=512, help='Codebook size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--eval-count', type=int, default=1000, help='Images used to compute FID/KID')
    parser.add_argument('--eval-seeds', type=int, default=3, help='Number of seeds for evaluation')
    args = parser.parse_args()

    env_flag = os.environ.get('USE_20_PERCENT', '')
    use_subset = args.use_20pct or (env_flag.lower() in ('1', 'true', 'yes'))

    dev = get_device()
    print(f'Using device: {dev}')

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
        train_loader = cfg['train_loader']
    else:
        print('Loading full ArtBench training split')
        root = Path(__file__).resolve().parent.parent
        train_hf, _ = load_artbench_train_split(root)
        transform = build_transform(image_size=32)
        train_ds = HFDatasetTorch(train_hf, transform=transform)
        train_loader = DataLoader(
            train_ds,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

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

    model = ConvVQVAE(embedding_dim=args.embedding_dim, num_embeddings=args.num_embeddings).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print('Starting VQ-VAE training on ArtBench training split...')
    history = train_vqvae(model, train_loader, optimizer, epochs=args.epochs)

    save_path = Path(__file__).resolve().parent / 'results'
    save_path.mkdir(exist_ok=True)
    with torch.no_grad():
        model.eval()
        samples = model.sample(100, device=dev)
        save_image(samples, save_path / 'vqvae_generated_samples.png', nrow=10)
        print(f'Saved VQ-VAE samples to {save_path}')

    feat_extractor = build_feature_extractor(dev)
    base_evaluation(None, 'baseline', test_loader, dev, feature_extractor=feat_extractor)
    metrics = evaluate_vqvae_metrics(
        model,
        test_loader,
        feat_extractor,
        eval_count=args.eval_count,
        num_seeds=args.eval_seeds,
    )
    print(
        f"VQ-VAE FID: {metrics['fid_mean']:.4f} +/- {metrics['fid_std']:.4f} | "
        f"KID: {metrics['kid_mean']:.4f} +/- {metrics['kid_std']:.4f}"
    )

    return model, history


if __name__ == '__main__':
    set_seed(42)
    main()