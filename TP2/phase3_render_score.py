"""
Phase 3 – Render & Score: generate images for all Phase 2 candidates and rank them.

For each target image, renders every candidate prompt with the fixed seed from
the filename, scores it against the target (CLIP sim, LPIPS, pixel MSE/RMSE),
and ranks the candidates under four criteria:

    rank_clip       – by CLIP image-image similarity   (higher better)
    rank_lpips      – by LPIPS perceptual distance      (lower  better)
    rank_mse        – by pixel MSE                       (lower  better)
    rank_composite  – Borda: average of the three ranks (lower  better)

The submitted top-3 per image uses rank_composite. The per-metric rankings are
kept too, so the report can show where the metrics agree / disagree.

Ranking by averaged ranks (Borda) needs no normalisation: each metric only
contributes an order, so the wildly different scales (CLIP ~0.5-0.95, LPIPS
~0.1-0.7, MSE ~0-20000) never get mixed as raw values.

Input:  phase2_candidates.json, target images in TARGET_DIR
Output: phase3_results.json   – every candidate, all metrics + all ranks
        phase3_top3.json/.csv – top-3 per image, deliverable format
        Renders under outputs/phase3/<image_stem>/candidate_NNN.png

Tip: already-rendered candidates are skipped (delete the PNG to re-render).
"""

import csv
import json
import re
import shutil
import statistics
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from tqdm import tqdm

from evaluation import evaluate_candidate


# ── Config ────────────────────────────────────────────────────────────────────

PHASE2_CANDIDATES = Path("phase2_candidates.json")
TARGET_DIR = Path("statement/TP2-students/students/tp2-chosen")
OUTPUT_DIR = Path("outputs/phase3")
TOP3_DIR = Path("outputs/phase3_top3")   # tidy copy of the winning renders
RESULTS_FILE = Path("phase3_results.json")
TOP3_JSON = Path("phase3_top3.json")
TOP3_CSV = Path("phase3_top3.csv")

TOP_K = 3  # prompts submitted per image

MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 8.0
LCM_ORIGIN_STEPS = 50
WIDTH = 768
HEIGHT = 768
FALLBACK_SEED = 2026

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def seed_from_filename(path: Path, fallback: int = FALLBACK_SEED) -> int:
    match = re.match(r"^(\d+)", path.stem)
    return int(match.group(1)) if match else fallback


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Pipeline ──────────────────────────────────────────────────────────────────

def load_pipeline(device: str):
    print(f"Loading {MODEL_ID} on {device} ...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe.to(device)
    return pipe


def render(prompt: str, seed: int, pipe, device: str) -> Image.Image:
    generator_device = "cpu" if device == "mps" else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        lcm_origin_steps=LCM_ORIGIN_STEPS,
        width=WIDTH,
        height=HEIGHT,
        output_type="pil",
        generator=generator,
    ).images[0]


# ── Ranking ───────────────────────────────────────────────────────────────────

def attach_ranks(rows: list[dict]) -> list[dict]:
    """Add rank_clip / rank_lpips / rank_mse / rank_composite, then sort by
    composite. Best rank = 1. Composite = mean of the three per-metric ranks."""
    n = len(rows)

    order_clip  = sorted(range(n), key=lambda i: rows[i]["clip_similarity"], reverse=True)
    order_lpips = sorted(range(n), key=lambda i: rows[i]["lpips"])
    order_mse   = sorted(range(n), key=lambda i: rows[i]["pixel_mse"])

    rank_clip  = {idx: r + 1 for r, idx in enumerate(order_clip)}
    rank_lpips = {idx: r + 1 for r, idx in enumerate(order_lpips)}
    rank_mse   = {idx: r + 1 for r, idx in enumerate(order_mse)}

    for i, row in enumerate(rows):
        row["rank_clip"]  = rank_clip[i]
        row["rank_lpips"] = rank_lpips[i]
        row["rank_mse"]   = rank_mse[i]
        row["rank_composite"] = round(
            (rank_clip[i] + rank_lpips[i] + rank_mse[i]) / 3, 3
        )

    rows.sort(key=lambda r: (r["rank_composite"], r["rank_clip"]))
    return rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def print_aggregate(label: str, rows: list[dict]) -> None:
    clip = [r["clip_similarity"] for r in rows]
    lp   = [r["lpips"] for r in rows]
    mse  = [r["pixel_mse"] for r in rows]
    cm, cs = mean_std(clip)
    lm, ls = mean_std(lp)
    mm, ms = mean_std(mse)
    print(f"  {label}  (n={len(rows)})")
    print(f"    CLIP  mean={cm:.4f}  std={cs:.4f}   (higher better)")
    print(f"    LPIPS mean={lm:.4f}  std={ls:.4f}   (lower better)")
    print(f"    MSE   mean={mm:.4f}  std={ms:.4f}   (lower better)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    candidates: dict[str, list[str]] = json.loads(PHASE2_CANDIDATES.read_text())
    target_by_name = {
        p.name: p
        for p in TARGET_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }

    device = default_device()
    print(f"Device: {device}")
    pipe = load_pipeline(device)

    all_results: dict[str, list[dict]] = {}

    for image_name, prompts in candidates.items():
        target_path = target_by_name.get(image_name)
        if target_path is None:
            print(f"Skipping unknown target: {image_name}")
            continue

        seed = seed_from_filename(target_path)
        print(f"\n── {image_name}  (seed={seed}, {len(prompts)} candidates) ──")

        out_dir = OUTPUT_DIR / Path(image_name).stem
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, prompt in enumerate(tqdm(prompts, desc=image_name), start=1):
            img_path = out_dir / f"candidate_{i:03d}.png"
            if not img_path.exists():
                render(prompt, seed, pipe, device).save(img_path)

            metrics = evaluate_candidate(target_path, img_path)
            rows.append({
                "target": str(target_path),
                "target_name": image_name,
                "render_seed": seed,
                "candidate_index": i,
                "prompt": prompt,
                "render": str(img_path),
                **metrics,
            })

        attach_ranks(rows)
        all_results[image_name] = rows

        print("  Top-3 by composite rank:")
        for r in rows[:TOP_K]:
            print(
                f"    [{r['candidate_index']:3d}] comp={r['rank_composite']:.2f}"
                f"  CLIP={r['clip_similarity']:.4f}(#{r['rank_clip']})"
                f"  LPIPS={r['lpips']:.4f}(#{r['rank_lpips']})"
                f"  MSE={r['pixel_mse']:.4f}(#{r['rank_mse']})"
            )

    # ── Persist full results ──────────────────────────────────────────────────
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved all candidates + ranks to {RESULTS_FILE}")

    # ── Deliverable: top-3 per image ──────────────────────────────────────────
    TOP3_DIR.mkdir(parents=True, exist_ok=True)
    top3_rows: list[dict] = []
    for image_name, rows in all_results.items():
        stem = Path(image_name).stem
        for rank, r in enumerate(rows[:TOP_K], start=1):
            # Copy the winning render into a tidy, self-documenting filename.
            dest = TOP3_DIR / f"{stem}_rank{rank}_candidate{r['candidate_index']:03d}.png"
            shutil.copyfile(r["render"], dest)
            top3_rows.append({
                **r,
                "submission_rank": rank,
                "top3_render": str(dest),
            })

    TOP3_JSON.write_text(json.dumps(top3_rows, indent=2, ensure_ascii=False))

    fields = [
        "target", "target_name", "render_seed", "submission_rank",
        "candidate_index", "prompt", "render", "top3_render",
        "clip_similarity", "lpips", "pixel_mse", "pixel_rmse",
        "rank_clip", "rank_lpips", "rank_mse", "rank_composite",
    ]
    with TOP3_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(top3_rows)
    print(f"Saved top-{TOP_K} deliverable to {TOP3_JSON} and {TOP3_CSV}")
    print(f"Copied {len(top3_rows)} winning renders to {TOP3_DIR}/")

    # ── Aggregate stats across the test set ───────────────────────────────────
    best1 = [rows[0] for rows in all_results.values()]
    print("\n" + "=" * 70)
    print("AGGREGATE METRICS ACROSS THE TEST SET")
    print("=" * 70)
    print_aggregate("Best prompt per image (top-1)", best1)
    print_aggregate(f"Submitted prompts (top-{TOP_K})", top3_rows)


if __name__ == "__main__":
    main()
