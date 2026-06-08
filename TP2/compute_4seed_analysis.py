#!/usr/bin/env python3
"""Pool the 4 completed optimiser-seed runs (1234,3456,5678,9012) and compute
every statistic / figure the report's Analysis section needs."""
import json, shutil, math
from pathlib import Path
import numpy as np

SEEDS = ["1234", "3456", "5678", "9012"]
IMAGES = ["1159_25.png", "1159_29.png", "1159_3.png", "1159_7.png", "7836.png", "9338.png"]
SHORT = {"1159_25.png": "orange juice", "1159_29.png": "palm tree", "1159_3.png": "fire character",
         "1159_7.png": "hedgehog", "7836.png": "astronaut", "9338.png": "rainbow hamster"}
BRANCHES = ["clip", "lpips", "mse", "composite"]
ROOT = Path(".")
FIGDIR = Path("report/figures"); GRIDDIR = FIGDIR / "grid"
GRIDDIR.mkdir(parents=True, exist_ok=True)

# ---- load all candidates from the 4 runs -------------------------------------
# per_img[img] = list of candidate dicts, each tagged with its seed
per_img = {im: [] for im in IMAGES}
for s in SEEDS:
    d = json.load(open(f"phase4_s{s}_results.json"))
    for im in IMAGES:
        for c in d[im]:
            c = dict(c); c["seed"] = s
            per_img[im].append(c)

all_cands = [c for im in IMAGES for c in per_img[im]]
print("total pooled candidates:", len(all_cands))
for im in IMAGES:
    print(f"  {im}: {len(per_img[im])}")

def avg_rank(values, ascending=True):
    """Average ranks, 1=best. ascending=True -> lower value is best."""
    arr = np.array(values, float)
    order = arr if ascending else -arr
    # rank with ties -> average
    idx = np.argsort(order, kind="mergesort")
    ranks = np.empty(len(arr), float)
    sorted_vals = order[idx]
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = r
        i = j + 1
    return ranks

# ---- composite rank, computed WITHIN each (image,seed) pool ------------------
for im in IMAGES:
    for s in SEEDS:
        pool = [c for c in per_img[im] if c["seed"] == s]
        if not pool:
            continue
        rc = avg_rank([c["clip_similarity"] for c in pool], ascending=False)  # higher clip best
        rl = avg_rank([c["lpips"] for c in pool], ascending=True)
        rm = avg_rank([c["pixel_mse"] for c in pool], ascending=True)
        for c, a, b, d in zip(pool, rc, rl, rm):
            c["rank_composite"] = round((a + b + d) / 3.0, 3)

# ---- metric direction helpers ------------------------------------------------
BEST = {"clip": ("clip_similarity", max), "lpips": ("lpips", min),
        "mse": ("pixel_mse", min), "composite": ("rank_composite", min)}

def stats(vals):
    a = np.array(vals, float)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=0)),
                min=float(a.min()), max=float(a.max()), n=int(a.size))

out = {"seeds": SEEDS, "n_total": len(all_cands), "per_image_counts": {im: len(per_img[im]) for im in IMAGES}}

# ===== (1) aggregate over all candidates, best-per-image, submitted-top3 ======
out["all_candidates"] = {m: stats([c[m] for c in all_cands])
                         for m in ["clip_similarity", "lpips", "pixel_mse", "pixel_rmse"]}

# best per image (global best on each metric, across all seeds)  -> n=6
best_clip = {im: max(per_img[im], key=lambda c: c["clip_similarity"]) for im in IMAGES}
best_lpips = {im: min(per_img[im], key=lambda c: c["lpips"]) for im in IMAGES}
best_mse = {im: min(per_img[im], key=lambda c: c["pixel_mse"]) for im in IMAGES}
out["best_per_image"] = {
    "clip_similarity": stats([best_clip[im]["clip_similarity"] for im in IMAGES]),
    "lpips": stats([best_lpips[im]["lpips"] for im in IMAGES]),
    "pixel_mse": stats([best_mse[im]["pixel_mse"] for im in IMAGES]),
}

# ===== (2) champion per (image, branch): best within branch on branch metric ==
# stored per (image,branch,seed) then reduced across seeds for the grid
def branch_pool(im, br, s=None):
    p = [c for c in per_img[im] if c["branch"] == br]
    if s is not None:
        p = [c for c in p if c["seed"] == s]
    return p

champ_seed = {}  # (im,br,seed) -> champion candidate
for im in IMAGES:
    for br in BRANCHES:
        key, fn = BEST[br]
        for s in SEEDS:
            p = branch_pool(im, br, s)
            if p:
                champ_seed[(im, br, s)] = fn(p, key=lambda c: c[key])

# grid champion = best across seeds for (image,branch)
grid = {}  # (im,br) -> champion across seeds
for im in IMAGES:
    for br in BRANCHES:
        key, fn = BEST[br]
        cands = [champ_seed[(im, br, s)] for s in SEEDS if (im, br, s) in champ_seed]
        grid[(im, br)] = fn(cands, key=lambda c: c[key])

# per-branch table: mean+/-std of metrics over the 6x4 per-seed champions
out["per_branch"] = {}
for br in BRANCHES:
    chs = [champ_seed[(im, br, s)] for im in IMAGES for s in SEEDS if (im, br, s) in champ_seed]
    out["per_branch"][br] = {
        "n": len(chs),
        "clip_similarity": stats([c["clip_similarity"] for c in chs]),
        "lpips": stats([c["lpips"] for c in chs]),
        "pixel_rmse": stats([c["pixel_rmse"] for c in chs]),
        "pixel_mse": stats([c["pixel_mse"] for c in chs]),
        "rank_composite": stats([c["rank_composite"] for c in chs]),
    }

# submitted top-3 = grid clip/lpips/mse cells (exclude composite) -> n=18
top3 = []
for im in IMAGES:
    for br in ["clip", "lpips", "mse"]:
        top3.append((im, br, grid[(im, br)]))
out["submitted_top3"] = {
    "clip_similarity": stats([c["clip_similarity"] for _, _, c in top3]),
    "lpips": stats([c["lpips"] for _, _, c in top3]),
    "pixel_mse": stats([c["pixel_mse"] for _, _, c in top3]),
    "pixel_rmse": stats([c["pixel_rmse"] for _, _, c in top3]),
    "n": len(top3),
}

# ===== (3) Spearman correlations over the pooled candidate set ================
def spearman(x, y):
    rx = avg_rank(x, ascending=True); ry = avg_rank(y, ascending=True)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (math.sqrt((rx @ rx) * (ry @ ry))))
clip = [c["clip_similarity"] for c in all_cands]
lp = [c["lpips"] for c in all_cands]
ms = [c["pixel_mse"] for c in all_cands]
out["spearman"] = {"clip_vs_lpips": spearman(clip, lp),
                   "clip_vs_mse": spearman(clip, ms),
                   "lpips_vs_mse": spearman(lp, ms)}

# ===== (4) per-seed best-per-image (stability across seeds) ===================
out["per_seed_best"] = {}
for s in SEEDS:
    bc = [max([c for c in per_img[im] if c["seed"] == s], key=lambda c: c["clip_similarity"])["clip_similarity"] for im in IMAGES]
    bl = [min([c for c in per_img[im] if c["seed"] == s], key=lambda c: c["lpips"])["lpips"] for im in IMAGES]
    bm = [min([c for c in per_img[im] if c["seed"] == s], key=lambda c: c["pixel_mse"])["pixel_mse"] for im in IMAGES]
    out["per_seed_best"][s] = {"clip": stats(bc), "lpips": stats(bl), "pixel_mse": stats(bm)}

# ===== copy grid images + targets =============================================
manifest = {}
for im in IMAGES:
    stem = im.replace(".png", "")
    tgt = ROOT / "statement/TP2-students/students/tp2-chosen" / im
    shutil.copy(tgt, GRIDDIR / f"{stem}__target.png")
    for br in BRANCHES:
        c = grid[(im, br)]
        dst = GRIDDIR / f"{stem}__{br}.png"
        shutil.copy(ROOT / c["render"], dst)
        manifest[f"{stem}__{br}"] = {k: c[k] for k in
            ["prompt", "clip_similarity", "lpips", "pixel_mse", "pixel_rmse", "rank_composite", "branch", "seed", "render"]}
out["grid_manifest"] = manifest

json.dump(out, open("report/analysis_4seed.json", "w"), indent=1)
print("\nwrote report/analysis_4seed.json and", len(list(GRIDDIR.glob('*.png'))), "grid images")
