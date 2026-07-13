#!/usr/bin/env python3
import json, shutil, csv
from pathlib import Path
import numpy as np

SEEDS = ["1234", "3456", "5678", "9012", "7890"]
IMAGES = ["1159_25.png", "1159_29.png", "1159_3.png", "1159_7.png", "7836.png", "9338.png"]
SHORT = {"1159_25.png": "orange juice", "1159_29.png": "palm tree", "1159_3.png": "fire character",
         "1159_7.png": "hedgehog", "7836.png": "astronaut", "9338.png": "rainbow hamster"}
TGTDIR = Path("statement/TP2-students/students/tp2-chosen")
FIGB = Path("report/figures/annexB"); FIGB.mkdir(parents=True, exist_ok=True)
DELV = Path("delivery/top3/seed_7890"); DELV.mkdir(parents=True, exist_ok=True)

def load_pool(seed, img):
    if seed == "7890":
        return json.loads(Path("phase4_s7890_checkpoint.json").read_text())["pool_by_image"][img]
    return json.load(open(f"phase4_s{seed}_results.json"))[img]

def avg_rank(vals, asc=True):
    a = np.array(vals, float); order = a if asc else -a
    idx = np.argsort(order, kind="mergesort"); r = np.empty(len(a)); sv = order[idx]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sv[j + 1] == sv[i]:
            j += 1
        rr = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[idx[k]] = rr
        i = j + 1
    return r

def ranked(pool):
    rc = avg_rank([c["clip_similarity"] for c in pool], asc=False)
    rl = avg_rank([c["lpips"] for c in pool], asc=True)
    rm = avg_rank([c["pixel_mse"] for c in pool], asc=True)
    comp = (rc + rl + rm) / 3.0
    out = []
    for c, x in zip(pool, comp):
        c = dict(c); c["composite"] = round(float(x), 3); out.append(c)
    out.sort(key=lambda c: c["composite"])
    return out

# ---- Annex B: top-1 per (seed, target) + grid images -------------------------
top1 = {}      # (seed,img) -> candidate
for seed in SEEDS:
    for img in IMAGES:
        best = ranked(load_pool(seed, img))[0]
        top1[(seed, img)] = best
        shutil.copy(best["render"], FIGB / f"s{seed}__{img}")
for img in IMAGES:                          # target reference row
    shutil.copy(TGTDIR / img, FIGB / f"target__{img}")

# ---- seed 7890 Borda top-3 -> delivery ---------------------------------------
rows7890 = []
for img in IMAGES:
    stem = img.replace(".png", "")
    for k, c in enumerate(ranked(load_pool("7890", img))[:3], 1):
        dst = DELV / f"{stem}_rank{k}_borda_cand{c['candidate_index']:03d}.png"
        shutil.copy(c["render"], dst)
        rows7890.append({"target": img, "render_seed": c["render_seed"], "submission_rank": k,
                         "branch": c["branch"], "iteration": c["iteration"],
                         "candidate_index": c["candidate_index"], "prompt": c["prompt"],
                         "image": dst.name, "clip_similarity": c["clip_similarity"],
                         "lpips": c["lpips"], "pixel_mse": c["pixel_mse"], "pixel_rmse": c["pixel_rmse"],
                         "rank_composite": c["composite"]})
with open(DELV / "top3.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows7890[0].keys())); w.writeheader(); w.writerows(rows7890)
Path(DELV / "top3.json").write_text(json.dumps(rows7890, indent=2, ensure_ascii=False))
print("seed_7890 delivery: 18 renders + csv/json written")

# ---- LaTeX for Annex B -------------------------------------------------------
def gi(name):
    return r"\includegraphics[width=0.150\textwidth]{figures/annexB/" + name + "}"
hdr = " & " + " & ".join(r"\scriptsize " + SHORT[im] for im in IMAGES) + r" \\[1pt]"
def grow(label, prefix):
    cells = " & ".join(gi(f"{prefix}__{im}") for im in IMAGES)
    return r"\rotatebox{90}{\scriptsize " + label + "} & " + cells + r" \\"
grid_rows = [hdr, grow("Target", "target")]
for seed in SEEDS:
    grid_rows.append(grow(f"seed {seed}", f"s{seed}"))
grid = "\n".join(grid_rows)

tbl = []
for img in IMAGES:
    for i, seed in enumerate(SEEDS):
        c = top1[(seed, img)]
        lab = (r"\multirow{5}{*}{\shortstack[l]{" + SHORT[img] + r"\\" + img.replace("_", r"\_") + "}}") if i == 0 else ""
        tbl.append(f"{lab} & {seed} & {c['branch']} & ${c['clip_similarity']:.3f}$ & ${c['pixel_rmse']:.3f}$ & ${c['lpips']:.3f}$ & ${c['composite']:.1f}$ \\\\")
    tbl.append(r"\midrule")
tbl[-1] = r"\bottomrule"
table = "\n".join(tbl)

Path("report/_annexB_grid.tex").write_text(grid)
Path("report/_annexB_table.tex").write_text(table)
print("wrote report/_annexB_grid.tex and report/_annexB_table.tex")
print("\n-- seed 7890 top-1 per target (sanity) --")
for img in IMAGES:
    c = top1[("7890", img)]
    print(f"  {img:12s} comp={c['composite']:.1f} clip={c['clip_similarity']:.3f} branch={c['branch']}")
