class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Add mean and log-variance heads from flattened conv features.
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 7, 7)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar


def vae_loss(xhat, x, mu, logvar, beta=0.7):
    # Reconstruction term: sum over pixels, then convert to per-sample value
    b = x.size(0)
    recon_sum = F.binary_cross_entropy(xhat, x, reduction='sum')
    # KL divergence (sum over latent dims), per-batch sum
    kl_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Convert to per-sample values for stable bookkeeping
    recon = recon_sum / b
    kl = kl_sum / b
    loss = recon + beta * kl
    return loss, recon, kl


def train_vae(model, loader, optimizer, epochs=20, beta=0.7):
    model.train()
    hist = []
    for ep in range(epochs):
        tl, tr, tk = 0.0, 0.0, 0.0
        for x, _ in tqdm(loader, leave=False):
            x = x.to(device)
            xhat, mu, logvar = model(x)

            b = x.size(0)
            # compute sums then convert to per-sample inside vae_loss
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item() * b
            tr += recon.item() * b
            tk += kl.item() * b
        n = len(loader.dataset)
        hist.append({'train_loss': tl / n, 'train_recon_bce': tr / n, 'train_kl': tk / n})
        print(f'Epoch {ep+1}/{epochs} | train_loss={tl/n:.4f} train_recon={tr/n:.4f} train_kl={tk/n:.4f}')
    return hist


def evaluate_vae(model, loader, beta=0.7):
    model.eval()
    tl, tr, tk, tm, ta, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            xhat, mu, logvar = model(x)
            b = x.size(0)

            # use sums internally and convert to per-sample in vae_loss
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)

            # accumulate reconstruction / kl totals and pixel metrics
            tl += loss.item() * b
            tr += recon.item() * b
            tk += kl.item() * b
            tm += F.mse_loss(xhat, x, reduction='sum').item()
            ta += F.l1_loss(xhat, x, reduction='sum').item()
            n += b
    numel = x[0].numel()
    return {'loss': tl / n, 'recon_bce': tr / n, 'kl': tk / n, 'mse': tm / (n * numel), 'mae': ta / (n * numel)}