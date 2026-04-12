import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image

# UPDATE 1: Import the new CCritic and train_cwgan_gp functions
# (Assuming you kept the file name as cDCGAN.py, update if you renamed it)
from cDCGAN import CGenerator, CCritic, train_cwgan_gp, init_weights

# Import from your provided files
from autoencoder import ConvVAE, train_vae
from data_loader import (
    setup_artbench_from_csv_subset,
    load_artbench_train_split,
    build_transform,
    HFDatasetTorch,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)
from diffusion import GaussianDiffusion, PixelUNet, train_diffusion

import numpy as np
from evaluation import (
    extract_inception_features,
    feature_statistics,
    frechet_distance,
    kid_score,
    evaluate_model_protocol,
    base_evaluation,
    build_feature_extractor,
    get_real_samples,
    generate_samples,
    generate_vae_samples_per_style,
)

from torchvision.models import inception_v3, Inception_V3_Weights

# Evaluation helpers moved to src/evaluation.py


def pipeline():
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Option: use whole training split or the 20% CSV subset
    # Set to True to run on the whole training split once. Set to False to use the CSV 20% subset.
    USE_FULL_DATASET = True

    root = Path(__file__).resolve().parent.parent
    if USE_FULL_DATASET:
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
    else:
        data_state = setup_artbench_from_csv_subset(project_root=root)
        train_loader = data_state['train_loader']
        class_names = data_state['class_names']

    num_classes = len(class_names)

    # 2. Hyperparameters
    LATENT_DIM = 128
    EPOCHS = 50
    LR = 1e-3
    BETA = 0.5
    
    # 3. Initialize Model 
    model = ConvVAE(latent_dim=LATENT_DIM, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training VAE
    print("Starting training on ArtBench 20% subset...")
    history = train_vae(
        model, 
        train_loader, 
        optimizer, 
        epochs=EPOCHS, 
        beta=BETA
    )

    model.eval()

    # UPDATE 2: Initialize CCritic instead of CDiscriminator
    gen = CGenerator(latent_dim=LATENT_DIM).to(device)
    critic = CCritic().to(device)
    gen.apply(init_weights)
    critic.apply(init_weights)

    # UPDATE 3: Call the new train_cwgan_gp function
    print("Starting cWGAN-GP training on 20% subset...")
    gan_history = train_cwgan_gp(gen, critic, train_loader, LATENT_DIM, epochs=EPOCHS)

    save_path = Path('results')
    save_path.mkdir(exist_ok=True)

    # UPDATE 4: Adjust plot to use 'c_loss' and update titles/filenames
    plt.figure(figsize=(8, 4))
    plt.plot(gan_history['c_loss'], label='Critic Loss')
    plt.plot(gan_history['g_loss'], label='Generator Loss')
    plt.title('cWGAN-GP Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / 'cwgan_gp_loss_curve.png')
    plt.close()

    with torch.no_grad():
        gen.eval()
        # Generate 10 examples for each of the 10 classes (styles) -> 100 images
        samples_per_style = 10
        num_styles = 10
        class_grid = torch.arange(num_styles, device=device).repeat_interleave(samples_per_style)
        noise = torch.randn(class_grid.size(0), LATENT_DIM, device=device)
        gan_samples = gen(noise, class_grid)
        # Already bounded to [0,1] with Sigmoid
        save_image(gan_samples, save_path / 'cwgan_gp_generated_samples.png', nrow=samples_per_style)
        print(f"Saved cWGAN-GP results to {save_path}")

    with torch.no_grad():
        vae_style_samples = generate_vae_samples_per_style(
            model=model,
            device=device,
            samples_per_style=10,
            num_styles=num_classes,
        )
        save_image(vae_style_samples, save_path / 'vae_generated_samples.png', nrow=10)
        print(f"Saved qualitative samples to {save_path}")

    # --- PART 2: DIFFUSION ---
    print("\n--- Starting Diffusion Phase ---")
    
    DIFF_TIMESTEPS = 1000
    DIFF_EPOCHS = 50
    DIFF_LR = 2e-4
    
    schedule = GaussianDiffusion(num_timesteps=DIFF_TIMESTEPS, device=device)
    diff_model = PixelUNet(in_channels=3, model_channels=64).to(device)

    print("Training Pixel-space Diffusion on 20% subset...")
    diff_history = train_diffusion(
        model=diff_model,
        loader=train_loader,
        schedule=schedule,
        epochs=DIFF_EPOCHS,
        lr=DIFF_LR
    )

    print("Generating diffusion samples...")
    diff_model.eval()
    with torch.no_grad():
        # Generate 10 examples per style (10 styles) => 100 samples.
        # NOTE: This diffusion model is unconditional in this repo, so
        # samples cannot be explicitly conditioned on style labels.
        # We still produce 100 samples and arrange them as a 10x10 grid.
        total_samples = 10 * 10
        samples = schedule.p_sample_loop(diff_model, shape=(total_samples, 3, 32, 32))
        samples = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)
        save_image(samples, save_path / 'diffusion_generated_samples.png', nrow=10)
        print(f"Saved diffusion samples to {save_path}")

    print("\n--- Starting Quantitative Evaluation Phase ---")
    feature_extractor = build_feature_extractor(device)

    base_evaluation(None, 'baseline', train_loader, device, feature_extractor, latent_dim=LATENT_DIM)

    evaluate_model_protocol(model, 'vae', train_loader, device, feature_extractor, latent_dim=LATENT_DIM, num_classes=num_classes)
    evaluate_model_protocol(gen, 'gan', train_loader, device, feature_extractor, latent_dim=LATENT_DIM)
    evaluate_model_protocol(diff_model, 'diffusion', train_loader, device, feature_extractor, latent_dim=LATENT_DIM)

if __name__ == "__main__":
    pipeline()