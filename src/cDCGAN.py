import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torchvision.utils import save_image

try:
    import optuna
except ImportError:
    optuna = None

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
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Kept BatchNorm in Generator, as it is safe and helps with generation
        self.net = nn.Sequential(
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
            nn.Tanh(),
        )

    def forward(self, z, labels):
        le = self.label_emb(labels).view(z.size(0), z.size(1), 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = torch.cat([z, le], dim=1)
        return self.net(x)

class CCritic(nn.Module): # Renamed to Critic
    def __init__(self, image_channels=3, num_classes=10, ndf=64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 32 * 32)
        
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(image_channels + 1, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)),
            # Removed Sigmoid here to output raw linear scores
        )

    def forward(self, x, labels):
        le = self.label_emb(labels).view(-1, 1, 32, 32)
        x = torch.cat([x, le], dim=1)
        return self.net(x).view(-1, 1)

def init_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname or 'InstanceNorm' in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# --- GRADIENT PENALTY ---

def compute_gradient_penalty(critic, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, labels)
    
    # Fake tensor for grad_outputs
    fake = torch.ones((real_samples.size(0), 1), device=device)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- TRAINING ---

def train_cwgan_gp(generator, critic, loader, latent_dim, epochs=20, lr=1e-4, n_critic=5, lambda_gp=10):
    # Updated optimizer betas for WGAN-GP
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_c = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

    history = {'g_loss': [], 'c_loss': []}
    
    for epoch in range(epochs):
        g_running, c_running, c_batches, g_batches = 0.0, 0.0, 0, 0
        
        for i, (real_imgs, labels, _) in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}')):
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            bs = real_imgs.size(0)

            # ---------------------
            #  Train Critic
            # ---------------------
            opt_c.zero_grad()
            
            # Generate fake images
            z = torch.randn(bs, latent_dim, device=device)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=device)
            fake_imgs = generator(z, gen_labels)

            # Real and Fake scores
            c_real = critic(real_imgs, labels)
            c_fake = critic(fake_imgs.detach(), gen_labels)

            # Gradient Penalty
            gradient_penalty = compute_gradient_penalty(critic, real_imgs.data, fake_imgs.data, labels.data)
            
            # Critic Loss = Fake - Real + Penalty (We want to maximize Real and minimize Fake)
            c_loss = torch.mean(c_fake) - torch.mean(c_real) + lambda_gp * gradient_penalty
            
            c_loss.backward()
            opt_c.step()
            
            c_running += c_loss.item()
            c_batches += 1

            # ---------------------
            #  Train Generator
            # ---------------------
            # Update Generator every n_critic iterations
            if i % n_critic == 0:
                opt_g.zero_grad()
                
                # We generated new fake images here to avoid using stale computational graphs
                z = torch.randn(bs, latent_dim, device=device)
                fake_imgs = generator(z, gen_labels)
                
                c_g_fake = critic(fake_imgs, gen_labels)
                
                # Generator Loss = -Fake (We want Critic to think fakes are real)
                g_loss = -torch.mean(c_g_fake)
                
                g_loss.backward()
                opt_g.step()
                
                g_running += g_loss.item()
                g_batches += 1

        history['g_loss'].append(g_running / g_batches if g_batches > 0 else 0)
        history['c_loss'].append(c_running / c_batches)
        print(f"Epoch {epoch+1} | Critic Loss: {history['c_loss'][-1]:.4f} | G Loss: {history['g_loss'][-1]:.4f}")

    return history

# --- INFERENCE & VISUALIZATION ---
# (Unchanged from original implementation)

@torch.no_grad()
def run_inference(generator, latent_dim, n_samples=10, specific_class=None):
    generator.eval()
    z = torch.randn(n_samples, latent_dim, device=device)
    if specific_class is not None:
        labels = torch.full((n_samples,), specific_class, dtype=torch.long, device=device)
    else:
        labels = torch.randint(0, NUM_CLASSES, (n_samples,), device=device)
    
    samples = generator(z, labels)
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


@torch.no_grad()
def evaluate_gan_metrics(
    generator,
    test_loader,
    feature_extractor,
    latent_dim,
    num_classes,
    eval_count=2000,
    num_seeds=3,
):
    """Evaluate GAN quality and return mean/std FID and KID for optimization."""
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
            num_classes=num_classes,
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


def run_cwgan_pipeline(
    train_loader,
    test_loader,
    num_classes,
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
    gen = CGenerator(latent_dim=latent_dim, num_classes=num_classes, ngf=base_channels).to(device)
    critic = CCritic(image_channels=IMG_CHANNELS, num_classes=num_classes, ndf=base_channels).to(device)
    gen.apply(init_weights)
    critic.apply(init_weights)

    print(
        f"[{run_name}] Starting cWGAN-GP training "
        f"(lr={learning_rate:.2e}, latent_dim={latent_dim}, base_channels={base_channels}, "
        f"base_updates_per_g={base_updates_per_g})"
    )
    history = train_cwgan_gp(
        gen,
        critic,
        train_loader,
        latent_dim,
        epochs=epochs,
        lr=learning_rate,
        n_critic=base_updates_per_g,
    )

    if save_samples:
        with torch.no_grad():
            gen.eval()
            class_grid = torch.arange(min(num_classes, 10), device=device).repeat_interleave(10)
            noise = torch.randn(class_grid.size(0), latent_dim, device=device)
            gan_samples = gen(noise, class_grid)
            gan_samples = torch.clamp((gan_samples + 1.0) / 2.0, 0.0, 1.0)
            save_image(gan_samples, save_path / f'{run_name}_generated_samples.png', nrow=10)

    metrics = evaluate_gan_metrics(
        gen,
        test_loader,
        feature_extractor,
        latent_dim=latent_dim,
        num_classes=num_classes,
        eval_count=eval_count,
        num_seeds=eval_seeds,
    )
    print(
        f"[{run_name}] FID: {metrics['fid_mean']:.4f} +/- {metrics['fid_std']:.4f} | "
        f"KID: {metrics['kid_mean']:.4f} +/- {metrics['kid_std']:.4f}"
    )

    return {
        'generator': gen,
        'critic': critic,
        'history': history,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-20pct', action='store_true', help='Use training_20_percent.csv subset')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per run/trial')
    parser.add_argument('--latent-dim', type=int, default=128, help='Latent dimensionality for non-search mode')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for non-search mode')
    parser.add_argument('--base-channels', type=int, default=64, choices=[32, 64, 128], help='Base channels (ngf/ndf) for non-search mode')
    parser.add_argument('--base-updates-per-g', type=int, default=5, choices=[3, 5, 7], help='Critic updates per generator update')
    parser.add_argument('--bayes-search', action='store_true', help='Run Bayesian hyperparameter search with Optuna (TPE sampler)')
    parser.add_argument('--n-trials', type=int, default=10, help='Number of Bayesian search trials')
    parser.add_argument('--eval-count', type=int, default=2000, help='Images used to compute FID/KID per trial')
    parser.add_argument('--eval-seeds', type=int, default=3, help='Number of random seeds per trial for metric averaging')
    args = parser.parse_args()
    env_flag = os.environ.get('USE_20_PERCENT', '')
    use_subset = args.use_20pct or (env_flag.lower() in ('1', 'true', 'yes'))

    # Device already set via get_device()/device
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
        # Load full ArtBench training split
        root = Path(__file__).resolve().parent.parent
        train_hf, class_names = load_artbench_train_split(root)
        transform = build_transform(image_size=32)
        train_ds = HFDatasetTorch(train_hf, transform=transform)
        train_loader = torch.utils.data.DataLoader(
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
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = len(class_names)

    # Output directory for samples and search summaries.
    save_path = Path(__file__).resolve().parent / 'results'
    save_path.mkdir(exist_ok=True)

    feat_extractor = build_feature_extractor(device)
    base_evaluation(
        None,
        'baseline',
        test_loader,
        device,
        feature_extractor=feat_extractor,
        latent_dim=args.latent_dim,
    )

    if args.bayes_search:
        if optuna is None:
            raise ImportError('Optuna is required for --bayes-search. Install with: pip install optuna')

        print('Starting Bayesian hyperparameter search for cWGAN-GP...')
        trial_dir = save_path / 'bayes_trials'
        trial_dir.mkdir(exist_ok=True)

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler, study_name='cdcgan_bayes_search')

        def objective(trial):
            latent_dim = trial.suggest_int('latent_dims', 64, 128, step=32)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            base_channels = trial.suggest_categorical('base_channels', [32, 64, 128])
            base_updates_per_g = trial.suggest_categorical('base_updates_per_g', [3, 5, 7])

            trial_name = f'trial_{trial.number:03d}'
            trial_result = run_cwgan_pipeline(
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=num_classes,
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

        best_params_path = save_path / 'cdcgan_bayes_best_params.json'
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

        # Retrain with the best hyperparameters and run the full evaluation protocol.
        best_result = run_cwgan_pipeline(
            train_loader=train_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            latent_dim=int(best_params['latent_dims']),
            learning_rate=float(best_params['learning_rate']),
            base_channels=int(best_params['base_channels']),
            base_updates_per_g=int(best_params['base_updates_per_g']),
            epochs=args.epochs,
            feature_extractor=feat_extractor,
            save_path=save_path,
            run_name='dcgan_best_bayes',
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
            num_classes=num_classes,
        )

        return best_result['generator'], best_result['critic'], best_result['history']

    default_result = run_cwgan_pipeline(
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        base_updates_per_g=args.base_updates_per_g,
        epochs=args.epochs,
        feature_extractor=feat_extractor,
        save_path=save_path,
        run_name='dcgan_generated_samples',
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
        num_classes=num_classes,
    )

    return default_result['generator'], default_result['critic'], default_result['history']


if __name__ == '__main__':
    main()