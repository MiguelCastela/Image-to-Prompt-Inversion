import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image

from data_loader import (
    load_artbench_train_split,
    load_artbench_splits,
    build_transform,
    HFDatasetTorch,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)
from data_loader import setup_artbench_from_csv_subset
import argparse
import os
from evaluation import build_feature_extractor, base_evaluation, evaluate_model_protocol

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            # input is Z: (N, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, img_channels, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def train_dcgan(generator, discriminator, loader, latent_dim=100, epochs=20, lr=2e-4, beta1=0.5):
    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    history = {'g_loss': [], 'd_loss': []}

    for epoch in range(epochs):
        g_running, d_running, d_batches, g_batches = 0.0, 0.0, 0, 0
        for real_imgs, _, _ in tqdm(loader, desc=f'Epoch {epoch+1}'):
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            # Train Discriminator
            discriminator.zero_grad()
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            real_out = discriminator(real_imgs)
            d_loss_real = criterion(real_out, real_labels)

            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = generator(z).detach()
            fake_out = discriminator(fake_imgs)
            d_loss_fake = criterion(fake_out, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            opt_d.step()

            d_running += d_loss.item()
            d_batches += 1

            # Train Generator
            generator.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            gen_imgs = generator(z)
            out = discriminator(gen_imgs)
            g_loss = criterion(out, real_labels)
            g_loss.backward()
            opt_g.step()

            g_running += g_loss.item()
            g_batches += 1

        history['d_loss'].append(d_running / d_batches if d_batches > 0 else 0)
        history['g_loss'].append(g_running / g_batches if g_batches > 0 else 0)
        print(f"Epoch {epoch+1} | D Loss: {history['d_loss'][-1]:.4f} | G Loss: {history['g_loss'][-1]:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-20pct', action='store_true', help='Use training_20_percent.csv subset')
    args = parser.parse_args()
    env_flag = os.environ.get('USE_20_PERCENT', '')
    use_subset = args.use_20pct or (env_flag.lower() in ('1', 'true', 'yes'))

    print(f"Using device: {device}")
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

    LATENT = 128
    EPOCHS = 50

    gen = Generator(latent_dim=LATENT).to(device)
    disc = Discriminator().to(device)
    gen.apply(weights_init_normal)
    disc.apply(weights_init_normal)

    print("Starting DCGAN (non-conditional) training...")
    history = train_dcgan(gen, disc, train_loader, latent_dim=LATENT, epochs=EPOCHS)

    # Save 100 non-conditional samples.
    save_path = Path(__file__).resolve().parent / 'results'
    save_path.mkdir(exist_ok=True)
    with torch.no_grad():
        gen.eval()
        n = 100
        z = torch.randn(n, LATENT, device=device)
        samples = gen(z)
        samples = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)
        save_image(samples, save_path / 'dcgan_nc_generated_samples.png', nrow=10)
        print(f"Saved DCGAN (non-conditional) samples to {save_path}")

    feat_extractor = build_feature_extractor(device)
    base_evaluation(None, 'baseline', test_loader, device, feature_extractor=feat_extractor, latent_dim=LATENT)
    evaluate_model_protocol(gen, 'gan', test_loader, device, feat_extractor, latent_dim=LATENT)

    return gen, disc, history


if __name__ == '__main__':
    main()
