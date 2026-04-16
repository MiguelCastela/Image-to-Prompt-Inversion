from scipy import linalg
import numpy as np

import torch
import torch.nn.functional as F

def prepare_images_for_inception(images, device):
    """
    Standardizes ArtBench RGB images for the Inception model.
    """
    # 1. Ensure images are on the correct device and float type
    images = images.to(device=device, dtype=torch.float32)

    # 1b. Convert loader/model outputs in [-1, 1] to [0, 1] when needed.
    if images.min() < 0.0:
        images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    
    # 2. Rescale/Interpolate to 299x299 (Inception requirement)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    # 3. Normalize using ImageNet statistics
    # ArtBench is RGB, so we use the 3-channel mean/std
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    return (images - mean) / std

@torch.no_grad()
def extract_inception_features(images, model, batch_size=32, device=None):
    """
    Passes images through the Inception extractor in batches.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval() # Ensure model is in evaluation mode
    features = []
    
    # Process in batches to avoid GPU memory overflow
    for start in range(0, len(images), batch_size):
        batch = prepare_images_for_inception(images[start : start + batch_size], device=device)
        feats = model(batch)
        
        # If the model has auxiliary outputs (standard Inception behavior), take the first
        if isinstance(feats, tuple):
            feats = feats[0]
            
        features.append(feats.detach().cpu())
        
    return torch.cat(features, dim=0)




import numpy as np

def feature_statistics(features):
    """
    Calculates mu and sigma for the feature distribution.
    """
    # Convert to float64 numpy for better precision in covariance calculation
    features = np.asarray(features, dtype=np.float64)
    
    # Calculate the average (mean) feature vector
    mu = features.mean(axis=0)
    
    # Calculate the covariance matrix (rowvar=False because rows are samples)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma
def polynomial_kernel(x, y):
    d = x.shape[1]
    return ((x @ y.T) / d + 1.0) ** 3


def kid_score(features_real, features_fake, subset_size=64, num_subsets=10, seed=123):
    x = np.asarray(features_real, dtype=np.float64)
    y = np.asarray(features_fake, dtype=np.float64)
    m = min(len(x), len(y), subset_size)
    if m < 2:
        raise ValueError('Need at least 2 samples per set to compute KID.')

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(num_subsets):
        x_idx = rng.choice(len(x), size=m, replace=False)
        y_idx = rng.choice(len(y), size=m, replace=False)
        x_sub = x[x_idx]
        y_sub = y[y_idx]

        k_xx = polynomial_kernel(x_sub, x_sub)
        k_yy = polynomial_kernel(y_sub, y_sub)
        k_xy = polynomial_kernel(x_sub, y_sub)

        mean_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
        mean_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        mean_xy = k_xy.mean()
        estimates.append(mean_xx + mean_yy - 2.0 * mean_xy)

    return float(np.mean(estimates)), float(np.std(estimates)) 
    



def frechet_distance(mu_r, sigma_r, mu_g, sigma_g, eps=1e-6):
    # 1) Convert inputs to float64 numpy arrays
    mu_r, sigma_r = np.atleast_1d(mu_r).astype(np.float64), np.atleast_2d(sigma_r).astype(np.float64)
    mu_g, sigma_g = np.atleast_1d(mu_g).astype(np.float64), np.atleast_2d(sigma_g).astype(np.float64)

    # 2) Compute diff = mu_r - mu_g
    diff = mu_r - mu_g

    # 3) Add eps * I for numerical stability
    offset = np.eye(sigma_r.shape[0]) * eps
    
    # 4) Compute the matrix square root of (Sigma_r)(Sigma_g)
    # Note: We use linalg.sqrtm from scipy
    cov_sqrt, _ = linalg.sqrtm((sigma_r + offset).dot(sigma_g + offset), disp=False)
    
    # 5) If the result is complex (due to noise), keep only the real part
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    # 6) Return the final FID scalar as a Python float
    fid = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * cov_sqrt)
    return float(fid)


# --- Higher-level evaluation utilities moved from main.py ---
from torchvision.models import inception_v3, Inception_V3_Weights


EVAL_CONFIG = {
    'reference_count': 5000,
    'generated_count': 5000,
    'num_seeds': 10,
    'kid_subsets': 50,
    'kid_subset_size': 100,
    'batch_size': 32,
}


def get_real_samples(loader, count=5000):
    images = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
        images.append(imgs)
        if len(torch.cat(images)) >= count:
            break
    return torch.cat(images)[:count]


def generate_samples(model, model_type, count=5000, latent_dim=128, device='cpu', schedule=None, num_classes=10):
    samples = []
    batch_size = 64
    model.eval()

    with torch.no_grad():
        while len(torch.cat(samples) if samples else []) < count:
            if model_type == 'vae':
                z = torch.randn(batch_size, latent_dim).to(device)
                labels = torch.randint(0, num_classes, (batch_size,), device=device)
                batch = model.decode(z, labels)
            elif model_type == 'gan':
                z = torch.randn(batch_size, latent_dim).to(device)
                labels = torch.randint(0, 10, (batch_size,)).to(device)
                try:
                    batch = model(z, labels)  # conditional GAN
                except TypeError:
                    batch = model(z)          # non-conditional GAN
                # GAN outputs are often tanh-scaled in [-1, 1]; map to [0, 1] for Inception.
                if batch.min() < 0.0:
                    batch = torch.clamp((batch + 1.0) / 2.0, 0.0, 1.0)
            elif model_type == 'diffusion':
                if schedule is None:
                    raise ValueError('Diffusion evaluation requires a valid schedule.')
                if hasattr(schedule, 'sample'):
                    batch = schedule.sample(model, shape=(batch_size, 3, 32, 32), sampler='ddim', sample_steps=100)
                else:
                    batch = schedule.p_sample_loop(model, shape=(batch_size, 3, 32, 32))
                batch = torch.clamp((batch + 1.0) / 2.0, 0.0, 1.0)

            samples.append(batch.cpu())

    return torch.cat(samples)[:count]


def generate_vae_samples_per_style(model, device, samples_per_style=10, num_styles=10):
    model.eval()
    generated_blocks = []
    with torch.no_grad():
        for style_id in range(num_styles):
            z = torch.randn(samples_per_style, model.latent_dim, device=device)
            labels = torch.full((samples_per_style,), style_id, dtype=torch.long, device=device)
            generated_blocks.append(model.decode(z, labels))

    return torch.cat(generated_blocks, dim=0)


def build_feature_extractor(device):
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def evaluate_model_protocol(model, model_type, loader, device, feature_extractor, latent_dim=128, num_classes=10, schedule=None):
    fid_scores = []
    kid_means = []

    # 1. Prepare Real Reference (Done once to save time)
    real_images = get_real_samples(loader, count=EVAL_CONFIG['reference_count'])
    real_features = extract_inception_features(real_images, feature_extractor, device=device)
    mu_real, sigma_real = feature_statistics(real_features)

    # 2. Statistical repetition loop
    for seed in range(EVAL_CONFIG['num_seeds']):
        torch.manual_seed(seed)
        print(f"  Evaluating seed {seed+1}/{EVAL_CONFIG['num_seeds']}...")

        # Generate samples
        gen_images = generate_samples(
            model,
            model_type,
            count=EVAL_CONFIG['generated_count'],
            latent_dim=latent_dim,
            device=device,
            schedule=schedule,
            num_classes=num_classes,
        )
        gen_features = extract_inception_features(gen_images, feature_extractor, device=device)

        # Compute FID
        mu_gen, sigma_gen = feature_statistics(gen_features)
        fid = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        fid_scores.append(fid)

        # Compute KID
        k_mean, _ = kid_score(real_features, gen_features, subset_size=EVAL_CONFIG['kid_subset_size'], num_subsets=EVAL_CONFIG['kid_subsets'])
        kid_means.append(k_mean)

    print(f"\nResults for {model_type.upper()} over {EVAL_CONFIG['num_seeds']} seeds:")
    print(f"  FID: {np.mean(fid_scores):.4f} ± {np.std(fid_scores):.4f}")
    print(f"  KID: {np.mean(kid_means):.4f} ± {np.std(kid_means):.4f}\n")


def base_evaluation(model, model_type, loader, device, feature_extractor=None, latent_dim=128):
    print("\n--- Starting Quantitative Evaluation Phase ---")

    if feature_extractor is None:
        feature_extractor = build_feature_extractor(device)
    batch_size = EVAL_CONFIG['batch_size']

    print("Running Baseline Sanity Checks...")
    real_eval_images = get_real_samples(loader, count=EVAL_CONFIG['reference_count'])
    noise_images = torch.rand(EVAL_CONFIG['reference_count'], 3, 32, 32)

    print("Extracting features (Real, Generated, Noise)...")
    feats_real = extract_inception_features(real_eval_images, feature_extractor, batch_size=batch_size)
    feats_noise = extract_inception_features(noise_images, feature_extractor, batch_size=batch_size)

    mid = len(feats_real) // 2
    real_a, real_b = feats_real[:mid], feats_real[mid:]
    noise_a = feats_noise[:mid]

    mu_ra, sigma_ra = feature_statistics(real_a)
    mu_rb, sigma_rb = feature_statistics(real_b)
    mu_na, sigma_na = feature_statistics(noise_a)

    fid_sanity = frechet_distance(mu_ra, sigma_ra, mu_rb, sigma_rb)
    fid_noise = frechet_distance(mu_ra, sigma_ra, mu_na, sigma_na)

    kid_sanity = kid_score(real_a, real_b, subset_size=EVAL_CONFIG['kid_subset_size'], num_subsets=EVAL_CONFIG['kid_subsets'])
    kid_noise = kid_score(real_a, noise_a, subset_size=EVAL_CONFIG['kid_subset_size'], num_subsets=EVAL_CONFIG['kid_subsets'])

    print(f"\n--- Evaluation Results (Protocol: {EVAL_CONFIG['reference_count']} images) ---")
    print(f"Sanity (Real vs Real): FID: {fid_sanity:.4f} | KID: {kid_sanity[0]:.4f} ± {kid_sanity[1]:.4f}")
    print(f"Baseline (Real vs Noise): FID: {fid_noise:.4f} | KID: {kid_noise[0]:.4f} ± {kid_noise[1]:.4f}")


