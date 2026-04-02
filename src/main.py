import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image
from cDCGAN import CGenerator, CDiscriminator, train_cgan, init_weights

# Import from your provided files
from autoencoder import ConvVAE, train_vae
from data_loader import setup_artbench_from_csv_subset
from diffusion import GaussianDiffusion, PixelUNet, train_diffusion

import numpy as np
from evaluation import extract_inception_features, feature_statistics, frechet_distance, kid_score


from torchvision.models import inception_v3, Inception_V3_Weights


EVAL_CONFIG = {
    'reference_count': 5000, 
    'generated_count': 5000,
    'kid_subsets': 50,
    'kid_subset_size': 100,
    'batch_size': 32
}


def evaluate_model_protocol(model, model_type, loader, device, feature_extractor, latent_dim=128, num_classes=10):


    fid_scores = []
    kid_means = []
    
    # 1. Prepare Real Reference (Done once to save time)
    real_images = get_real_samples(loader, count=5000) # Defined in previous response
    real_features = extract_inception_features(real_images, feature_extractor, device=device)
    mu_real, sigma_real = feature_statistics(real_features)

    # 2. Statistical Repetition Loop (Mandatory: 10 times )
    for seed in range(10):
        torch.manual_seed(seed)
        print(f"  Evaluating seed {seed+1}/10...")
        
        # Generate 5,000 samples 
        gen_images = generate_samples(
            model,
            model_type,
            count=5000,
            latent_dim=latent_dim,
            device=device,
            num_classes=num_classes,
        )
        gen_features = extract_inception_features(gen_images, feature_extractor, device=device)
        
        # Compute FID [cite: 74, 179]
        mu_gen, sigma_gen = feature_statistics(gen_features)
        fid = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        fid_scores.append(fid)
        
        # Compute KID (50 subsets of 100 [cite: 77, 181])
        k_mean, _ = kid_score(real_features, gen_features, subset_size=100, num_subsets=50)
        kid_means.append(k_mean)

    print(f"\nResults for {model_type.upper()} over 10 seeds:")
    print(f"  FID: {np.mean(fid_scores):.4f} ± {np.std(fid_scores):.4f}")
    print(f"  KID: {np.mean(kid_means):.4f} ± {np.std(kid_means):.4f}\n")



def base_evaluation(model, model_type, loader, device, feature_extractor, latent_dim=128):


    print("\n--- Starting Quantitative Evaluation Phase ---")

    feature_extractor = build_feature_extractor(device)
    # 1) Set extraction batch size
    batch_size = EVAL_CONFIG['batch_size']


    print("Running Baseline Sanity Checks...")
    real_eval_images = get_real_samples(loader, count=5000)
    noise_images = torch.rand(5000, 3, 32, 32) # Standard RGB noise baseline


    
    # 2) Extract features
    print("Extracting features (Real, Generated, Noise)...")
    # Note: Use extract_inception_features from the notebook context
    feats_real = extract_inception_features(real_eval_images, feature_extractor, batch_size=batch_size)
    feats_noise = extract_inception_features(noise_images, feature_extractor, batch_size=batch_size)

    # 3) Split real features for sanity check (Real A vs Real B)
    mid = len(feats_real) // 2
    real_a, real_b = feats_real[:mid], feats_real[mid:]

    # 4) Align Generated and Noise counts with the Real halves for consistency
    noise_a = feats_noise[:mid]

    # 5) Compute statistics (mu and sigma) for all groups
    mu_ra, sigma_ra = feature_statistics(real_a)
    mu_rb, sigma_rb = feature_statistics(real_b)
    mu_na, sigma_na = feature_statistics(noise_a)

    # 6) Compute FID comparisons
    fid_sanity = frechet_distance(mu_ra, sigma_ra, mu_rb, sigma_rb)
    fid_noise = frechet_distance(mu_ra, sigma_ra, mu_na, sigma_na)

    # 7) Compute KID comparisons (using enunciado's 50 subsets of 100) 
    kid_sanity = kid_score(real_a, real_b, subset_size=EVAL_CONFIG['kid_subset_size'], num_subsets=EVAL_CONFIG['kid_subsets'])
    kid_noise = kid_score(real_a, noise_a, subset_size=EVAL_CONFIG['kid_subset_size'], num_subsets=EVAL_CONFIG['kid_subsets'])

    # 8) Print results
    print(f"\n--- Evaluation Results (Protocol: {EVAL_CONFIG['reference_count']} images) ---")
    print(f"Sanity (Real vs Real): FID: {fid_sanity:.4f} | KID: {kid_sanity[0]:.4f} ± {kid_sanity[1]:.4f}")
    print(f"Baseline (Real vs Noise): FID: {fid_noise:.4f} | KID: {kid_noise[0]:.4f} ± {kid_noise[1]:.4f}")


def build_feature_extractor(device):
    """Initializes the Inception-v3 model for feature extraction."""
    # Load pretrained Inception-v3 
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    # Remove the final pooling/FC layer to get raw 2048-dim features
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def get_real_samples(loader, count=5000):
    """Extract real images from the data loader."""
    images = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]  # Handle (image, label) tuples
        else:
            imgs = batch
        images.append(imgs)
        if len(torch.cat(images)) >= count:
            break
    return torch.cat(images)[:count]

def generate_samples(model, model_type, count=5000, latent_dim=128, device='cpu', schedule=None, num_classes=10):
    """Generate samples from the model using appropriate family logic."""
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
                batch = model(z, labels)
            elif model_type == 'diffusion':
                # Diffusion needs the scheduler's reverse loop 
                batch = schedule.p_sample_loop(model, shape=(batch_size, 3, 32, 32))
                batch = (batch + 1.0) / 2.0  # Denormalize to [0, 1]
            
            samples.append(batch.cpu())
            
    return torch.cat(samples)[:count]


def generate_vae_samples_per_style(model, device, samples_per_style=10, num_styles=10):
    """Generate conditional VAE samples for each style id in [0, num_styles)."""
    model.eval()
    generated_blocks = []
    with torch.no_grad():
        for style_id in range(num_styles):
            z = torch.randn(samples_per_style, model.latent_dim, device=device)
            labels = torch.full((samples_per_style,), style_id, dtype=torch.long, device=device)
            generated_blocks.append(model.decode(z, labels))

    return torch.cat(generated_blocks, dim=0)

    



def pipeline():
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_state = setup_artbench_from_csv_subset(project_root=Path(__file__).resolve().parent.parent)
    train_loader = data_state['train_loader']
    class_names = data_state['class_names']
    num_classes = len(class_names)

    # 2. Hyperparameters (Project requirement: report these [cite: 26])
    LATENT_DIM = 128
    EPOCHS = 20
    LR = 1e-3
    BETA = 0.5  # VAE regularization weight
    
    # 3. Initialize Model 
    # Note: Ensure you updated autoencoder.py to handle 3 channels and 32x32 size
    model = ConvVAE(latent_dim=LATENT_DIM, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training on the 20% Subset
    # The project dictates using this subset for fast iteration [cite: 58, 193]
    print("Starting training on ArtBench 20% subset...")
    history = train_vae(
        model, 
        train_loader, 
        optimizer, 
        epochs=EPOCHS, 
        beta=BETA
    )

    # 5. Qualitative Evaluation: Generate Samples
    # Generative models must produce plausible samples from the distribution [cite: 34]
    model.eval()




    gen = CGenerator(latent_dim=LATENT_DIM).to(device)
    disc = CDiscriminator().to(device)
    gen.apply(init_weights)
    disc.apply(init_weights)



    print("Starting GAN training on 20% subset...")
    gan_history = train_cgan(gen, disc, train_loader, LATENT_DIM, epochs=EPOCHS)

    # Save cDCGAN training curves and generated samples to match other model outputs.
    save_path = Path('results')
    save_path.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(gan_history['d_loss'], label='Discriminator Loss')
    plt.plot(gan_history['g_loss'], label='Generator Loss')
    plt.title('cDCGAN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / 'cdcgan_loss_curve.png')
    plt.close()

    with torch.no_grad():
        gen.eval()
        class_grid = torch.arange(10, device=device).repeat_interleave(8)
        noise = torch.randn(class_grid.size(0), LATENT_DIM, device=device)
        gan_samples = gen(noise, class_grid)
        gan_samples = (gan_samples + 1.0) / 2.0
        save_image(gan_samples, save_path / 'cdcgan_generated_samples.png', nrow=10)
        print(f"Saved cDCGAN results to {save_path}")

    with torch.no_grad():
        vae_style_samples = generate_vae_samples_per_style(
            model=model,
            device=device,
            samples_per_style=10,
            num_styles=num_classes,
        )

        save_image(vae_style_samples, save_path / 'vae_generated_samples.png', nrow=10)
        print(f"Saved qualitative samples to {save_path}")

    # --- PART 2: DIFFUSION (New) ---
    print("\n--- Starting Diffusion Phase ---")
    
    # 1. Hyperparameters (Requirement: report these [cite: 26])
    DIFF_TIMESTEPS = 1000
    DIFF_EPOCHS = 20 
    DIFF_LR = 2e-4
    
    # 2. Initialize Scheduler and Model
    # Important: in_channels=3 for ArtBench-10 RGB 
    schedule = GaussianDiffusion(num_timesteps=DIFF_TIMESTEPS, device=device)
    diff_model = PixelUNet(in_channels=3, model_channels=64).to(device)

    # 3. Training on 20% Subset [cite: 58]
    print("Training Pixel-space Diffusion on 20% subset...")
    diff_history = train_diffusion(
        model=diff_model,
        loader=train_loader,
        schedule=schedule,
        epochs=DIFF_EPOCHS,
        lr=DIFF_LR
    )

    # 4. Qualitative Evaluation: Sampling
    print("Generating diffusion samples...")
    diff_model.eval()
    with torch.no_grad():
        # Generate 16 samples starting from pure noise
        # shape = [batch_size, channels, height, width]
        samples = schedule.p_sample_loop(diff_model, shape=(16, 3, 32, 32))
        
        # Denormalize from [-1, 1] to [0, 1] for saving
        samples = (samples + 1.0) / 2.0
        save_image(samples, save_path / 'diffusion_generated_samples.png', nrow=4)
        print(f"Saved diffusion samples to {save_path}")



    print("\n--- Starting Quantitative Evaluation Phase ---")
    feature_extractor = build_feature_extractor(device)




    base_evaluation(model, 'vae', train_loader, device, feature_extractor, latent_dim=LATENT_DIM)
    base_evaluation(gen, 'gan', train_loader, device, feature_extractor, latent_dim=LATENT_DIM)
    base_evaluation(diff_model, 'diffusion', train_loader, device, feature_extractor, latent_dim=LATENT_DIM)


    evaluate_model_protocol(model, 'vae', train_loader, device, feature_extractor, latent_dim=LATENT_DIM, num_classes=num_classes)
    evaluate_model_protocol(gen, 'gan', train_loader, device, feature_extractor, latent_dim=LATENT_DIM)
    evaluate_model_protocol(diff_model, 'diffusion', train_loader, device, feature_extractor, latent_dim=LATENT_DIM)




if __name__ == "__main__":
    pipeline()