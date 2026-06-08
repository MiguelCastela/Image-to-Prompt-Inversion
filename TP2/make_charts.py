#!/usr/bin/env python3
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = ["1234", "3456", "5678", "9012"]
IMAGES = ["1159_25.png", "1159_29.png", "1159_3.png", "1159_7.png", "7836.png", "9338.png"]
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

# reload pooled candidates
per_img = {im: [] for im in IMAGES}
for s in SEEDS:
    d = json.load(open(f"phase4_s{s}_results.json"))
    for im in IMAGES:
        for c in d[im]:
            c = dict(c); c["seed"] = s; per_img[im].append(c)
allc = [c for im in IMAGES for c in per_img[im]]
clip = np.array([c["clip_similarity"] for c in allc])
lp = np.array([c["lpips"] for c in allc])
ms = np.array([c["pixel_mse"] for c in allc])

# ---- Figure 1: correlation scatter -------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
ax[0].scatter(clip, lp, s=3, alpha=0.10, color="#1f4e79", edgecolors="none")
ax[0].set_xlabel("CLIP similarity"); ax[0].set_ylabel("LPIPS")
ax[0].set_title(r"CLIP vs LPIPS  ($\rho=-0.27$)", fontsize=9)
ax[1].scatter(lp, ms, s=3, alpha=0.10, color="#7a3b1f", edgecolors="none")
ax[1].set_xlabel("LPIPS"); ax[1].set_ylabel("pixel MSE")
ax[1].set_title(r"LPIPS vs MSE  ($\rho=+0.74$)", fontsize=9)
fig.tight_layout()
fig.savefig("report/figures/corr_scatter.png", dpi=170)
print("wrote report/figures/corr_scatter.png")

# ---- Figure 2: per-seed best-per-image stability -----------------------------
def best_per_image(seed, key, fn):
    return [fn([c for c in per_img[im] if c["seed"] == seed], key=lambda c: c[key])[key] for im in IMAGES]

metrics = [("CLIP similarity $\\uparrow$", "clip_similarity", max),
           ("LPIPS $\\downarrow$", "lpips", min),
           ("pixel MSE $\\downarrow$", "pixel_mse", min)]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
x = np.arange(len(SEEDS))
for ax, (title, key, fn) in zip(axes, metrics):
    means = [np.mean(best_per_image(s, key, fn)) for s in SEEDS]
    stds = [np.std(best_per_image(s, key, fn)) for s in SEEDS]
    ax.bar(x, means, yerr=stds, capsize=3, color="#4a6fa5", width=0.6,
           error_kw=dict(lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels(SEEDS, rotation=0, fontsize=7.5)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("optimiser seed", fontsize=8)
fig.tight_layout()
fig.savefig("report/figures/seed_stability.png", dpi=170)
print("wrote report/figures/seed_stability.png")
