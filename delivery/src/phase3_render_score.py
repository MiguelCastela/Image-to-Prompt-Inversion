import csv
import json
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from tqdm import tqdm

from evaluation import evaluate_candidate

PHASE2_CANDIDATES = Path("phase2_candidates.json")
TARGET_DIR = Path("statement/TP2-students/students/tp2-chosen")
OUTPUT_DIR = Path("outputs/phase3")
TOP3_DIR = Path("outputs/phase3_top3")
RESULTS_FILE = Path("phase3_results.json")
TOP3_JSON = Path("phase3_top3.json")
TOP3_CSV = Path("phase3_top3.csv")
SUMMARY_FILE = Path("phase3_summary.json")

TOP_K = 3

MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 8.0
LCM_ORIGIN_STEPS = 50
WIDTH = 768
HEIGHT = 768
FALLBACK_SEED = 2026

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def seed_from_filename(path: Path, fallback: int = FALLBACK_SEED) -> int:
    match = re.match(r"^(\d+)", path.stem)
    return int(match.group(1)) if match else fallback

def load_negative_prompt() -> str:
    try:
        import clip_interrogator
    except ImportError:
        print("  ! clip-interrogator not installed — rendering with no negative prompt.")
        return ""
    data_dir = Path(clip_interrogator.__file__).parent / "data"
    lines = [l.strip() for l in (data_dir / "negative.txt").read_text().splitlines()
             if l.strip()]
    return ", ".join(lines)

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

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

def render(
    prompt: str, seed: int, pipe, device: str, negative_prompt: str = ""
) -> Image.Image:
    generator_device = "cpu" if device == "mps" else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    return pipe(
        prompt=prompt,

        negative_prompt=negative_prompt or None,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,

        original_inference_steps=LCM_ORIGIN_STEPS,
        width=WIDTH,
        height=HEIGHT,
        output_type="pil",
        generator=generator,
    ).images[0]

def attach_ranks(rows: list[dict]) -> list[dict]:
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
    print(f"    CLIP      mean={cm:.4f}  std={cs:.4f}   (higher better)")
    print(f"    LPIPS     mean={lm:.4f}  std={ls:.4f}   (lower better)")
    print(f"    MSE       mean={mm:.4f}  std={ms:.4f}   (lower better)")
    if rows and "rank_composite" in rows[0]:
        comp = [r["rank_composite"] for r in rows]
        km, ks = mean_std(comp)
        print(f"    Composite mean={km:.4f}  std={ks:.4f}   (Borda rank, lower better)")

def _stats(values: list[float]) -> dict:
    m, s = mean_std(values)
    return {
        "mean": round(m, 6),
        "std": round(s, 6),
        "min": round(min(values), 6) if values else 0.0,
        "max": round(max(values), 6) if values else 0.0,
        "n": len(values),
    }

def _metric_block(rows: list[dict]) -> dict:
    block = {
        "clip_similarity": _stats([r["clip_similarity"] for r in rows]),
        "lpips": _stats([r["lpips"] for r in rows]),
        "pixel_mse": _stats([r["pixel_mse"] for r in rows]),
    }
    if rows and "rank_composite" in rows[0]:
        block["rank_composite"] = _stats([r["rank_composite"] for r in rows])
    return block

def _spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        rk = [0.0] * n
        for pos, idx in enumerate(order):
            rk[idx] = pos + 1
        return rk

    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n))
    vb = sum((rb[i] - mb) ** 2 for i in range(n))
    return round(cov / (va * vb) ** 0.5, 4) if va and vb else 0.0

def build_summary(
    all_results: dict[str, list[dict]],
    top3_rows: list[dict],
    device: str,
    dtype: str,
) -> dict:
    all_rows = [r for rows in all_results.values() for r in rows]
    best1 = [rows[0] for rows in all_results.values()]

    clip = [r["clip_similarity"] for r in all_rows]
    lp = [r["lpips"] for r in all_rows]
    mse = [r["pixel_mse"] for r in all_rows]
    correlations = {
        "clip_vs_lpips": _spearman(clip, lp),
        "clip_vs_mse": _spearman(clip, mse),
        "lpips_vs_mse": _spearman(lp, mse),
    }

    per_image = {}
    for name, rows in all_results.items():
        top = rows[0]
        per_image[name] = {
            "render_seed": top["render_seed"],
            "n_candidates": len(rows),
            "best_candidate_index": top["candidate_index"],
            "best_prompt": top["prompt"],
            "best_clip_similarity": round(top["clip_similarity"], 6),
            "best_lpips": round(top["lpips"], 6),
            "best_pixel_mse": round(top["pixel_mse"], 6),
            "metrics": _metric_block(rows),
        }

    return {
        "metadata": {
            "model_id": MODEL_ID,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "original_inference_steps": LCM_ORIGIN_STEPS,
            "width": WIDTH,
            "height": HEIGHT,
            "device": device,
            "dtype": dtype,
            "seed_policy": "leading digits of target filename (fallback 2026)",
            "top_k": TOP_K,
            "n_images": len(all_results),
            "n_candidates_total": len(all_rows),
            "metrics": ["clip_similarity", "lpips", "pixel_mse", "rank_composite"],
            "ranking": "composite = mean(rank_clip, rank_lpips, rank_mse), lower better",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "aggregate": {
            "all_candidates": _metric_block(all_rows),
            "best_per_image_top1": _metric_block(best1),
            "submitted_top3": _metric_block(top3_rows),
        },
        "metric_rank_correlations_spearman": correlations,
        "per_image": per_image,
    }

def main():
    candidates: dict[str, list[str]] = json.loads(PHASE2_CANDIDATES.read_text())
    target_by_name = {
        p.name: p
        for p in TARGET_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }

    device = default_device()
    print(f"Device: {device}")
    negative = load_negative_prompt()
    print(f"Negative prompt: {negative}")
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
                render(prompt, seed, pipe, device, negative_prompt=negative).save(img_path)

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

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved all candidates + ranks to {RESULTS_FILE}")

    TOP3_DIR.mkdir(parents=True, exist_ok=True)
    top3_rows: list[dict] = []
    for image_name, rows in all_results.items():
        stem = Path(image_name).stem
        for rank, r in enumerate(rows[:TOP_K], start=1):

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

    best1 = [rows[0] for rows in all_results.values()]
    print("\n" + "=" * 70)
    print("AGGREGATE METRICS ACROSS THE TEST SET")
    print("=" * 70)
    print_aggregate("Best prompt per image (top-1)", best1)
    print_aggregate(f"Submitted prompts (top-{TOP_K})", top3_rows)

    dtype = "float16" if device == "cuda" else "float32"
    summary = build_summary(all_results, top3_rows, device, dtype)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved report summary (aggregates, correlations, metadata) to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
