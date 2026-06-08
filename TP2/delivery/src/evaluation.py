import numpy as np
import torch
from PIL import Image
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

try:
    import lpips as lpips_lib
    _lpips_available = True
except ImportError:
    _lpips_available = False

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

_clip_model = None
_clip_processor = None

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"Loading CLIP model {CLIP_MODEL_ID} ...")
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _clip_model.eval()
    return _clip_model, _clip_processor

def compute_clip_similarity(image_a: Image.Image, image_b: Image.Image) -> float:
    model, processor = _get_clip()
    inputs = processor(images=[image_a, image_b], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    features = features / features.norm(dim=-1, keepdim=True)
    return float(features[0] @ features[1])

_lpips_fn = None

def _get_lpips():
    global _lpips_fn
    if not _lpips_available:
        raise ImportError(
            "lpips is not installed. Run: pip install lpips"
        )
    if _lpips_fn is None:
        print("Loading LPIPS (AlexNet) ...")
        _lpips_fn = lpips_lib.LPIPS(net="alex")
        _lpips_fn.eval()
    return _lpips_fn

def _pil_to_lpips_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def compute_lpips(image_a: Image.Image, image_b: Image.Image) -> float:
    fn = _get_lpips()
    ta = _pil_to_lpips_tensor(image_a)
    tb = _pil_to_lpips_tensor(image_b)
    with torch.no_grad():
        dist = fn(ta, tb)
    return float(dist)

def compute_pixel_mse(image_a: Image.Image, image_b: Image.Image) -> float:
    a = np.array(image_a.convert("RGB"), dtype=np.float32) / 255.0
    b = np.array(image_b.convert("RGB"), dtype=np.float32) / 255.0
    return float(np.mean((a - b) ** 2))

def compute_pixel_rmse(image_a: Image.Image, image_b: Image.Image) -> float:
    a = np.array(image_a.convert("RGB"), dtype=np.float32) / 255.0
    b = np.array(image_b.convert("RGB"), dtype=np.float32) / 255.0
    return float(np.sqrt(np.mean((a - b) ** 2)))

def evaluate_candidate(target_path, generated_path) -> dict:
    target = Image.open(target_path).convert("RGB")
    generated = Image.open(generated_path).convert("RGB")
    return {
        "clip_similarity": compute_clip_similarity(target, generated),
        "lpips": compute_lpips(target, generated),
        "pixel_mse": compute_pixel_mse(target, generated),
        "pixel_rmse": compute_pixel_rmse(target, generated),
    }

def evaluate_rows(rows: list[dict]) -> list[dict]:
    results = []
    for row in rows:
        metrics = evaluate_candidate(row["target"], row["render"])
        results.append({**row, **metrics})
    return results

def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("No rows to summarise.")
        return

    print(f"\n{'target':<20} {'cand':>4}  {'CLIP':>7}  {'LPIPS':>7}  {'RMSE':>7}  prompt")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['target_name']:<20} {row['candidate_index']:>4}"
            f"  {row['clip_similarity']:>7.4f}"
            f"  {row['lpips']:>7.4f}"
            f"  {row['pixel_rmse']:>7.2f}"
            f"  {row['prompt']}"
        )

    clips = [r["clip_similarity"] for r in rows]
    lpipss = [r["lpips"] for r in rows]
    rmses = [r["pixel_rmse"] for r in rows]
    print("-" * 80)
    print(
        f"{'mean':<26}"
        f"  {np.mean(clips):>7.4f}"
        f"  {np.mean(lpipss):>7.4f}"
        f"  {np.mean(rmses):>7.2f}"
    )
    print(
        f"{'std':<26}"
        f"  {np.std(clips):>7.4f}"
        f"  {np.std(lpipss):>7.4f}"
        f"  {np.std(rmses):>7.2f}"
    )

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate a TP2 output run.")
    parser.add_argument("run_json", help="Path to a generated_prompts.json file.")
    args = parser.parse_args()

    with open(args.run_json) as f:
        rows = json.load(f)

    evaluated = evaluate_rows(rows)
    print_summary(evaluated)

    out_path = Path(args.run_json).with_name("evaluated_prompts.json")
    out_path.write_text(json.dumps(evaluated, indent=2, ensure_ascii=False))
    print(f"\nSaved evaluated results to {out_path}")
