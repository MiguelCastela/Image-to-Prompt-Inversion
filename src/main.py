import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image
from cDCGAN import CGenerator, CDiscriminator, train_cgan, init_weights

# Import from your provided files
from autoencoder import ConvVAE, train_vae
from data_loader import train_loader_from_csv, class_names
from diffusion import GaussianDiffusion, PixelUNet, train_diffusion

def main():
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Hyperparameters (Project requirement: report these [cite: 26])
    LATENT_DIM = 128
    EPOCHS = 20
    LR = 1e-3
    BETA = 0.5  # VAE regularization weight
    
    # 3. Initialize Model 
    # Note: Ensure you updated autoencoder.py to handle 3 channels and 32x32 size
    model = ConvVAE(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training on the 20% Subset
    # The project dictates using this subset for fast iteration [cite: 58, 193]
    print("Starting training on ArtBench 20% subset...")
    history = train_vae(
        model, 
        train_loader_from_csv, 
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
    gan_history = train_cgan(gen, disc, train_loader_from_csv, LATENT_DIM, epochs=EPOCHS)

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
        # Sample from standard normal distribution N(0, I) [cite: 150]
        z = torch.randn(64, LATENT_DIM).to(device)
        samples = model.decode(z)

        save_image(samples, save_path / 'vae_generated_samples.png', nrow=8)
        print(f"Saved qualitative samples to {save_path}")

    # 6. Quantitative Evaluation Protocol (Conceptual)
    # Project requires FID and KID with 5000 samples [cite: 70, 72, 170]
    print("\nEvaluation Protocol Reminder:")
    print("- Generate 5,000 samples for final evaluation[cite: 70].")
    print("- Compute FID (Full set) and KID (50 subsets of 100)[cite: 74, 77].")
    print("- Repeat 10 times with different random seeds for statistical reporting[cite: 80, 176].")


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
        loader=train_loader_from_csv,
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



if __name__ == "__main__":
    main()