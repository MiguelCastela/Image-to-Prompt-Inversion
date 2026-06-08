#!/usr/bin/env python3
import json
import numpy as np

SEEDS = ["1234", "3456", "5678", "9012"]
IMAGES = ["1159_25.png", "1159_29.png", "1159_3.png", "1159_7.png", "7836.png", "9338.png"]

per_img = {im: [] for im in IMAGES}
for s in SEEDS:
    d = json.load(open(f"phase4_s{s}_results.json"))
    for im in IMAGES:
        for c in d[im]:
            c = dict(c); c["seed"] = s; per_img[im].append(c)

def avg_rank(values, ascending=True):
    arr = np.array(values, float)
    order = arr if ascending else -arr
    idx = np.argsort(order, kind="mergesort")
    ranks = np.empty(len(arr), float); sv = order[idx]; i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and sv[j + 1] == sv[i]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = r
        i = j + 1
    return ranks

# composite rank within each (image,seed) pool, tag each candidate
for im in IMAGES:
    for s in SEEDS:
        pool = [c for c in per_img[im] if c["seed"] == s]
        if not pool:
            continue
        rc = avg_rank([c["clip_similarity"] for c in pool], ascending=False)
        rl = avg_rank([c["lpips"] for c in pool], ascending=True)
        rm = avg_rank([c["pixel_mse"] for c in pool], ascending=True)
        for c, a, b, e in zip(pool, rc, rl, rm):
            c["composite"] = (a + b + e) / 3.0

allc = [c for im in IMAGES for c in per_img[im]]

def st(vals, nd):
    a = np.array(vals, float)
    return f"{a.mean():.{nd}f}\\pm{a.std():.{nd}f}"

def row(name, clip, lp, ms, comp):
    return f"{name} & ${st(clip,3)}$ & ${st(lp,3)}$ & ${st(ms,4)}$ & ${st(comp,1)}$ \\\\"

# warm start: one per target (identical across seeds for clip/lpips/mse); composite avg over seeds
warm_clip, warm_lp, warm_ms, warm_comp = [], [], [], []
for im in IMAGES:
    warms = [c for c in per_img[im] if c["branch"] == "warm"]
    warm_clip.append(warms[0]["clip_similarity"])
    warm_lp.append(warms[0]["lpips"])
    warm_ms.append(warms[0]["pixel_mse"])
    warm_comp.append(np.mean([w["composite"] for w in warms]))

# all candidates
ac = ([c["clip_similarity"] for c in allc], [c["lpips"] for c in allc],
      [c["pixel_mse"] for c in allc], [c["composite"] for c in allc])

# best per image (best value of each metric per target, across all seeds) n=6
bp_clip = [max(c["clip_similarity"] for c in per_img[im]) for im in IMAGES]
bp_lp = [min(c["lpips"] for c in per_img[im]) for im in IMAGES]
bp_ms = [min(c["pixel_mse"] for c in per_img[im]) for im in IMAGES]
bp_comp = [min(c["composite"] for c in per_img[im]) for im in IMAGES]

# submitted top-3 = grid clip/lpips/mse champions (best within branch on own metric, across seeds)
BEST = {"clip": ("clip_similarity", max), "lpips": ("lpips", min), "mse": ("pixel_mse", min)}
t3 = []
for im in IMAGES:
    for br in ["clip", "lpips", "mse"]:
        key, fn = BEST[br]
        # per-seed champion then best across seeds
        champs = []
        for s in SEEDS:
            p = [c for c in per_img[im] if c["branch"] == br and c["seed"] == s]
            if p:
                champs.append(fn(p, key=lambda c: c[key]))
        t3.append(fn(champs, key=lambda c: c[key]))
t3_t = ([c["clip_similarity"] for c in t3], [c["lpips"] for c in t3],
        [c["pixel_mse"] for c in t3], [c["composite"] for c in t3])

print("% --- replacement body rows for Table 2 ---")
print(row(r"Warm start ($n{=}6$)", warm_clip, warm_lp, warm_ms, warm_comp))
print(r"\midrule")
print(row(r"All candidates ($n{=}5{,}551$)", *ac))
# best per image bold
print(f"Best per image ($n{{=}}6$) & $\\mathbf{{{st(bp_clip,3)}}}$ & $\\mathbf{{{st(bp_lp,3)}}}$ & $\\mathbf{{{st(bp_ms,4)}}}$ & $\\mathbf{{{st(bp_comp,1)}}}$ \\\\")
print(row(r"Submitted top-3 ($n{=}18$)", *t3_t))
