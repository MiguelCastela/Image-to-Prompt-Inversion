import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
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
from evaluation import (
    build_feature_extractor,
    base_evaluation,
    evaluate_model_protocol,
    extract_inception_features,
    feature_statistics,
    frechet_distance,
    kid_score,
    get_real_samples,
    generate_samples,
)

try:
    import optuna
except ImportError:
    optuna = None

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


def train_dcgan(generator, discriminator, loader, latent_dim=100, epochs=20, lr=2e-4, beta1=0.5, n_critic=1):
    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    history = {'g_loss': [], 'd_loss': []}

    for epoch in range(epochs):
        g_running, d_running, d_batches, g_batches = 0.0, 0.0, 0, 0
        for step_idx, (real_imgs, _, _) in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}')):
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

            # Train Generator every n_critic discriminator updates.
            if step_idx % n_critic == 0:
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


@torch.no_grad()
def evaluate_gan_metrics(
    generator,
    test_loader,
    feature_extractor,
    latent_dim,
    eval_count=2000,
    num_seeds=3,
):
    fid_scores = []
    kid_scores = []

    real_images = get_real_samples(test_loader, count=eval_count)
    real_features = extract_inception_features(real_images, feature_extractor, device=device)
    mu_real, sigma_real = feature_statistics(real_features)

    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        gen_images = generate_samples(
            generator,
            model_type='gan',
            count=eval_count,
            latent_dim=latent_dim,
            device=device,
            num_classes=10,
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


def run_dcgan_pipeline(
    train_loader,
    test_loader,
    latent_dim,
    learning_rate,
    base_channels,
    base_updates_per_g,
    epochs,
    feature_extractor,
    save_path,
    run_name,
    eval_count=2000,
    eval_seeds=3,
    save_samples=True,
):
    gen = Generator(latent_dim=latent_dim, ngf=base_channels).to(device)
    disc = Discriminator(ndf=base_channels).to(device)
    gen.apply(weights_init_normal)
    disc.apply(weights_init_normal)

    print(
        f"[{run_name}] Starting DCGAN training "
        f"(lr={learning_rate:.2e}, latent_dim={latent_dim}, base_channels={base_channels}, "
        f"base_updates_per_g={base_updates_per_g})"
    )
    history = train_dcgan(
        gen,
        disc,
        train_loader,
        latent_dim=latent_dim,
        epochs=epochs,
        lr=learning_rate,
        n_critic=base_updates_per_g,
    )

    if save_samples:
        with torch.no_grad():
            gen.eval()
            n = 100
            z = torch.randn(n, latent_dim, device=device)
            samples = gen(z)
            samples = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)
            save_image(samples, save_path / f'{run_name}_generated_samples.png', nrow=10)

    metrics = evaluate_gan_metrics(
        gen,
        test_loader,
        feature_extractor,
        latent_dim=latent_dim,
        eval_count=eval_count,
        num_seeds=eval_seeds,
    )
    print(
        f"[{run_name}] FID: {metrics['fid_mean']:.4f} +/- {metrics['fid_std']:.4f} | "
        f"KID: {metrics['kid_mean']:.4f} +/- {metrics['kid_std']:.4f}"
    )

    return {
        'generator': gen,
        'discriminator': disc,
        'history': history,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-20pct', action='store_true', help='Use training_20_percent.csv subset')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per run/trial')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimensionality for non-search mode')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate for non-search mode')
    parser.add_argument('--base-channels', type=int, default=64, choices=[32, 64, 128], help='Base channels (ngf/ndf) for non-search mode')
    parser.add_argument('--base-updates-per-g', type=int, default=5, choices=[3, 5, 7], help='Discriminator updates per generator update')
    parser.add_argument('--bayes-search', action='store_true', help='Run Bayesian hyperparameter search with Optuna (TPE sampler)')
    parser.add_argument('--n-trials', type=int, default=10, help='Number of Bayesian search trials')
    parser.add_argument('--eval-count', type=int, default=2000, help='Images used to compute FID/KID per trial')
    parser.add_argument('--eval-seeds', type=int, default=3, help='Number of random seeds per trial for metric averaging')
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

    save_path = Path(__file__).resolve().parent / 'results'
    save_path.mkdir(exist_ok=True)

    feat_extractor = build_feature_extractor(device)
    base_evaluation(None, 'baseline', test_loader, device, feature_extractor=feat_extractor, latent_dim=args.latent_dim)

    if args.bayes_search:
        if optuna is None:
            raise ImportError('Optuna is required for --bayes-search. Install with: pip install optuna')

        print('Starting Bayesian hyperparameter search for DCGAN...')
        trial_dir = save_path / 'bayes_trials_dcgan'
        trial_dir.mkdir(exist_ok=True)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler, study_name='dcgan_bayes_search')

        def objective(trial):
            latent_dim = trial.suggest_int('latent_dims', 64, 128, step=32)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            base_channels = trial.suggest_categorical('base_channels', [32, 64, 128])
            base_updates_per_g = trial.suggest_categorical('base_updates_per_g', [3, 5, 7])

            trial_name = f'trial_{trial.number:03d}'
            trial_result = run_dcgan_pipeline(
                train_loader=train_loader,
                test_loader=test_loader,
                latent_dim=latent_dim,
                learning_rate=learning_rate,
                base_channels=base_channels,
                base_updates_per_g=base_updates_per_g,
                epochs=args.epochs,
                feature_extractor=feat_extractor,
                save_path=trial_dir,
                run_name=trial_name,
                eval_count=args.eval_count,
                eval_seeds=args.eval_seeds,
                save_samples=True,
            )

            trial.set_user_attr('kid_mean', trial_result['kid_mean'])
            trial.set_user_attr('kid_std', trial_result['kid_std'])
            return trial_result['fid_mean']

        study.optimize(objective, n_trials=args.n_trials)

        best = study.best_trial
        best_params = best.params
        print(f"Best trial #{best.number} -> FID: {best.value:.4f}")
        print(f"Best params: {best_params}")

        best_params_path = save_path / 'dcgan_bayes_best_params.json'
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

        best_result = run_dcgan_pipeline(
            train_loader=train_loader,
            test_loader=test_loader,
            latent_dim=int(best_params['latent_dims']),
            learning_rate=float(best_params['learning_rate']),
            base_channels=int(best_params['base_channels']),
            base_updates_per_g=int(best_params['base_updates_per_g']),
            epochs=args.epochs,
            feature_extractor=feat_extractor,
            save_path=save_path,
            run_name='dcgan_nc_best_bayes',
            eval_count=args.eval_count,
            eval_seeds=args.eval_seeds,
            save_samples=True,
        )

        evaluate_model_protocol(
            best_result['generator'],
            'gan',
            test_loader,
            device,
            feat_extractor,
            latent_dim=int(best_params['latent_dims']),
        )
        return best_result['generator'], best_result['discriminator'], best_result['history']

    default_result = run_dcgan_pipeline(
        train_loader=train_loader,
        test_loader=test_loader,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        base_updates_per_g=args.base_updates_per_g,
        epochs=args.epochs,
        feature_extractor=feat_extractor,
        save_path=save_path,
        run_name='dcgan_nc',
        eval_count=args.eval_count,
        eval_seeds=args.eval_seeds,
        save_samples=True,
    )

    evaluate_model_protocol(
        default_result['generator'],
        'gan',
        test_loader,
        device,
        feat_extractor,
        latent_dim=args.latent_dim,
    )

    return default_result['generator'], default_result['discriminator'], default_result['history']


if __name__ == '__main__':
    main()
