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


