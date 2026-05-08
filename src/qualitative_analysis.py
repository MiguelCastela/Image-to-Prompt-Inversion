import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
import seaborn as sns

# Import model components and evaluation tools
from cDiffusion import GaussianDiffusion, PixelUNet
from evaluation import build_feature_extractor, extract_inception_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def slerp(val, low, high):
    """
    Spherical Linear Interpolation for PyTorch Tensors.
    """
    # Normalize inputs for calculating angle
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    
    # Dot product gives cos(theta)
    dot = (low_norm * high_norm).sum(dim=1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    # Avoid division by zero
    if (so.abs() < 1e-5).all():
        return (1.0 - val) * low + val * high
        
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high

def lerp(val, low, high):
    """
    Simple Linear Interpolation.
    """
    return (1.0 - val) * low + val * high

def load_cdiffusion_model(ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    config = ckpt['config']
    
    diff_model = PixelUNet(
        in_channels=3,
        model_channels=config['base_channels'],
        num_classes=ckpt['num_classes'],
        use_attention=config['use_attention'],
        num_res_blocks=config['num_res_blocks'],
    ).to(device)
    
    schedule = GaussianDiffusion(
        num_timesteps=config['timesteps'], 
        device=device, 
        beta_schedule=config.get('beta_schedule', 'linear')
    )
    
    # Load EMA models directly to the inferencing model
    diff_model.load_state_dict(ckpt['model_state_dict'])
    diff_model.eval()
    
    # Using inference params
    sample_config = {
        'cfg_scale': config.get('cfg_scale', 6.0),
        'sampler': config.get('sampler', 'ddim'),
        'sample_steps': config.get('sample_steps', 100),
    }
    
    return diff_model, schedule, sample_config

@torch.no_grad()
def generate_interpolations(model, schedule, config, class_id=0, num_steps=8, mode='slerp'):
    print(f"Generating {mode.upper()} interpolation for class {class_id}...")
    shape = (1, 3, 32, 32)
    
    # Standard normal source and target noise
    z1 = torch.randn(shape, device=device)
    z2 = torch.randn(shape, device=device)
    
    alphas = torch.linspace(0, 1, num_steps, device=device)
    interpolated_samples = []
    
    labels = torch.tensor([class_id], device=device, dtype=torch.long)
    
    for alpha in tqdm(alphas, desc=f"{mode.upper()} steps"):
        alpha_val = alpha.item()
        
        # Flatten noise vectors for interpolation calculation
        z1_flat = z1.view(1, -1)
        z2_flat = z2.view(1, -1)
        
        if mode == 'slerp':
            z_interp_flat = slerp(alpha, z1_flat, z2_flat)
        else:
            z_interp_flat = lerp(alpha, z1_flat, z2_flat)
            
        z_interp = z_interp_flat.view(shape)
        
        # Feed the interpolated starting noise sequentially into reverse diffusion
        sample = schedule.sample(
            model,
            shape,
            labels=labels,
            guidance_scale=config['cfg_scale'],
            sampler=config['sampler'],
            sample_steps=config['sample_steps'],
            initial_noise=z_interp
        )
        
        # Transform [-1, 1] pixel values to [0, 1] range plotting format
        sample = (sample.clamp(-1, 1) + 1.0) / 2.0
        interpolated_samples.append(sample.cpu())

    return torch.cat(interpolated_samples, dim=0)

@torch.no_grad()
def generate_tsne(model, schedule, config, num_samples_per_class=50):
    print(f"Generating samples for t-SNE plot ({num_samples_per_class} per class)...")
    inception_model = build_feature_extractor(device)
    inception_model.eval()
    
    all_features = []
    all_labels = []
    num_classes = 10
    
    for c in range(num_classes):
        print(f"Sampling class {c}...")
        # Sample in small batches so we don't OOM
        batch_size = min(25, num_samples_per_class)
        generated_for_class = []
        for _ in range(0, num_samples_per_class, batch_size):
            shape = (batch_size, 3, 32, 32)
            labels = torch.full((batch_size,), c, device=device, dtype=torch.long)
            
            samples = schedule.sample(
                model,
                shape,
                labels=labels,
                guidance_scale=config['cfg_scale'],
                sampler=config['sampler'],
                sample_steps=config['sample_steps']
            )
            # Transform from [-1, 1] generating limits to [0, 1] feature input targets
            samples = (samples.clamp(-1, 1) + 1.0) / 2.0
            generated_for_class.append(samples)
            
        generated_for_class = torch.cat(generated_for_class, dim=0)
        
        # Pass conditional generation data through evaluated inception feature extractor
        features = extract_inception_features(generated_for_class, inception_model, batch_size=32, device=device)
        all_features.append(features)
        
        labels_arr = np.full((features.shape[0],), c)
        all_labels.append(labels_arr)
        
    all_features_np = np.concatenate(all_features, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    
    print("Running t-SNE on Inception features... (reducing to 2 dimensions)")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
    tsne_results = tsne.fit_transform(all_features_np)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Add KDE density contours to better visualize cluster bodies with low sample counts
    sns.kdeplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=all_labels_np,
        palette=sns.color_palette("tab10", num_classes),
        fill=True,
        alpha=0.3,
        levels=5,
        warn_singular=False
    )

    # Overlay individual generation points
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=all_labels_np,
        palette=sns.color_palette("tab10", num_classes),
        legend="full",
        alpha=0.9,
        edgecolor="w",
        s=40
    )
    plt.title(f"t-SNE Density of Generated ArtBench-10 (cDiffusion, cfg={config['cfg_scale']})")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(script_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    save_path = os.path.join(res_dir, "cdiffusion_tsne_plot.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"Saved t-SNE plot to {save_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(script_dir, "results")
    ckpt_path = os.path.join(res_dir, "cdiffusion_final_checkpoint.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        exit(1)
        
    model, schedule, config = load_cdiffusion_model(ckpt_path)
    
    # Choose a class to interpolate (Artbench class indices 0-9)
    test_class = 3
    
    # 1. Run SLERP (Spherical Interpolation — ideal for high-d gaussian noise parameters)
    slerp_samples = generate_interpolations(model, schedule, config, class_id=test_class, num_steps=8, mode='slerp')
    grid = make_grid(slerp_samples, nrow=8, normalize=False)
    plt.figure(figsize=(15, 3))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f"cDiffusion SLERP Interpolation (Class {test_class})")
    plt.savefig(os.path.join(res_dir, "cdiffusion_slerp.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Run LERP (Simple Linear combination)
    lerp_samples = generate_interpolations(model, schedule, config, class_id=test_class, num_steps=8, mode='lerp')
    grid = make_grid(lerp_samples, nrow=8, normalize=False)
    plt.figure(figsize=(15, 3))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f"cDiffusion LERP Interpolation (Class {test_class})")
    plt.savefig(os.path.join(res_dir, "cdiffusion_lerp.png"), bbox_inches='tight')
    plt.close()

    # 3. Create TSNE mappings of output generated features
    generate_tsne(model, schedule, config, num_samples_per_class=200) # Generates 200x10=2000 images total
