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

# data loading and evaluation helpers
from data_loader import (
    load_artbench_train_split,
    build_transform,
    HFDatasetTorch,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)
from evaluation import build_feature_extractor, base_evaluation, evaluate_model_protocol

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
            nn.Sigmoid(), # Keeping Sigmoid if your real images are scaled [0, 1]
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
            nn.Conv2d(image_channels + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Swapped BatchNorm2d for InstanceNorm2d to allow Gradient Penalty to work
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
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


def main():
    # Device already set via get_device()/device
    print(f"Using device: {device}")

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

    num_classes = len(class_names)

    # Hyperparameters (match src/main.py)
    LATENT = 128
    EPOCHS = 50

    # Initialize models
    gen = CGenerator(latent_dim=LATENT, num_classes=num_classes).to(device)
    critic = CCritic().to(device)
    gen.apply(init_weights)
    critic.apply(init_weights)

    # Train cWGAN-GP
    print("Starting cWGAN-GP training (DCGAN file)...")
    gan_history = train_cwgan_gp(gen, critic, train_loader, LATENT, epochs=EPOCHS)

    # Save a quick grid of generated samples
    save_path = Path('results')
    save_path.mkdir(exist_ok=True)
    with torch.no_grad():
        gen.eval()
        class_grid = torch.arange(min(num_classes, 10), device=device).repeat_interleave(10)
        noise = torch.randn(class_grid.size(0), LATENT, device=device)
        gan_samples = gen(noise, class_grid)
        save_image(gan_samples, save_path / 'dcgan_generated_samples.png', nrow=10)
        print(f"Saved DCGAN samples to {save_path}")

    # --- Evaluation: FID & KID ---
    feat_extractor = build_feature_extractor(device)

    base_evaluation(None, 'baseline', train_loader, device, feature_extractor=feat_extractor, latent_dim=LATENT)

    evaluate_model_protocol(gen, 'gan', train_loader, device, feat_extractor, latent_dim=LATENT)

    return gen, critic, gan_history


if __name__ == '__main__':
    main()