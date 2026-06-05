"""
Phase 3 – Render & Score: generate images for all Phase 2 candidates and rank them.

For each target image, renders every candidate prompt with the fixed seed from
the filename, scores it against the target (CLIP sim, LPIPS, pixel RMSE),
and saves the ranked results.

Input:  phase2_candidates.json, target images in TARGET_DIR
Output: phase3_results.json  (per image: candidates sorted by CLIP sim desc)
        Rendered images saved under outputs/phase3/<image_stem>/candidate_NNN.png

Tip: already-rendered candidates are skipped (remove the file to re-render).
"""

import json
import re
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
RESULTS_FILE = Path("phase3_results.json")

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
                generated = render(prompt, seed, pipe, device)
                generated.save(img_path)

            metrics = evaluate_candidate(target_path, img_path)
            rows.append({
                "image": image_name,
                "candidate_index": i,
                "prompt": prompt,
                "render": str(img_path),
                **metrics,
            })

        rows.sort(key=lambda r: r["clip_similarity"], reverse=True)
        all_results[image_name] = rows

        print(f"  Top-3 by CLIP similarity:")
        for r in rows[:3]:
            print(
                f"    [{r['candidate_index']:3d}]"
                f"  CLIP={r['clip_similarity']:.4f}"
                f"  LPIPS={r['lpips']:.4f}"
                f"  RMSE={r['pixel_rmse']:.1f}"
                f"  | {r['prompt']}"
            )

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nSaved all results to {RESULTS_FILE}")

    print("\n" + "=" * 80)
    print("FINAL TOP-3 PROMPTS PER IMAGE (by CLIP similarity)")
    print("=" * 80)
    for image_name, rows in all_results.items():
        print(f"\n{image_name}:")
        for r in rows[:3]:
            print(
                f"  [{r['candidate_index']:3d}]"
                f"  CLIP={r['clip_similarity']:.4f}"
                f"  LPIPS={r['lpips']:.4f}"
                f"  RMSE={r['pixel_rmse']:.1f}"
                f"\n        {r['prompt']}"
            )


if __name__ == "__main__":
    main()
