"""
Phase 1 – Warm Start (CLIP branch): zero-shot component extraction.

Uses CLIP to score clip-interrogator vocabulary banks (artists, mediums,
art movements) plus hand-crafted lighting/camera-angle banks.

Subject description is handled by BLIP-2 (phase1_captions.json) — CLIP
is not used for subject, matching the clip-interrogator design.

Labels are kept by threshold rather than fixed top-k: all labels within
SCORE_THRESHOLD of the top score are retained (capped at MAX_RESULTS).
This captures genuine uncertainty (e.g. two mediums with near-equal scores)
without flooding the LLM with noise.

Model: openai/clip-vit-large-patch14  (same encoder used in evaluation.py)
Output: phase1_clip.json
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ── Config ─────────────────────────────────────────────────────────────────────

TARGET_DIR = Path("statement/TP2-students/students/tp2-chosen")
OUTPUT_FILE = Path("phase1_clip.json")

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SCORE_THRESHOLD = 0.04   # keep labels within this much of the top score
MAX_RESULTS = 10         # hard cap per component (guards against huge artist bank)
TEXT_BATCH_SIZE = 256    # texts per CLIP forward pass


# ── Vocabulary ─────────────────────────────────────────────────────────────────

def load_ci_vocab() -> dict[str, list[str]]:
    """Load vocabulary lists from the clip-interrogator package data folder."""
    try:
        import clip_interrogator
    except ImportError:
        raise ImportError("Run: pip install clip-interrogator")

    data_dir = Path(clip_interrogator.__file__).parent / "data"

    def read(fname: str) -> list[str]:
        p = data_dir / fname
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]

    artists   = read("artists.txt")    # 5 265 entries
    mediums   = read("mediums.txt")    #    95 entries  (prefixed "a …")
    movements = read("movements.txt")  #   200 entries
    flavors   = read("flavors.txt")    # 100 970 entries

    return {
        "medium":       mediums,
        "artist_style": artists + movements,
        "flavor":       flavors,
    }


# Hand-crafted banks for components absent from clip-interrogator
FIXED_BANKS: dict[str, list[str]] = {
    "lighting": [
        "soft natural lighting", "golden hour lighting", "studio lighting",
        "dramatic cinematic lighting", "volumetric lighting", "rim lighting",
        "neon lighting", "candlelight", "backlit", "ambient occlusion lighting",
        "harsh directional lighting", "moody atmospheric lighting",
        "bright even lighting", "dark dramatic shadows", "iridescent lighting",
    ],
    "camera_angle": [
        "extreme close-up shot", "close-up portrait shot", "medium shot",
        "full body shot", "wide angle shot", "bird's eye view", "low angle shot",
        "eye level shot", "overhead shot", "dutch angle shot",
        "macro photography shot", "establishing wide shot",
    ],
}


# ── Device ─────────────────────────────────────────────────────────────────────

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── CLIP helpers ───────────────────────────────────────────────────────────────

def load_clip(device: str):
    print(f"Loading {CLIP_MODEL_ID} on {device} ...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    model.eval()
    return model, processor


def encode_texts(
    model, processor, texts: list[str], device: str
) -> torch.Tensor:
    """Return normalised text embeddings, shape (N, D), stored on CPU."""
    all_feats = []
    for i in range(0, len(texts), TEXT_BATCH_SIZE):
        batch = texts[i : i + TEXT_BATCH_SIZE]
        inputs = processor(
            text=batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        all_feats.append(F.normalize(feats, dim=-1).cpu())
    return torch.cat(all_feats, dim=0)


def encode_image(
    model, processor, image: Image.Image, device: str
) -> torch.Tensor:
    """Return normalised image embedding, shape (1, D), on CPU."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    return F.normalize(feats, dim=-1).cpu()


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_and_threshold(
    image_feat: torch.Tensor,
    labels: list[str],
    text_feats: torch.Tensor,
    threshold: float = SCORE_THRESHOLD,
    max_results: int = MAX_RESULTS,
) -> dict:
    scores = (image_feat @ text_feats.T).squeeze(0).tolist()
    ranked = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    top_score = ranked[0][1]
    kept = [
        (label, score)
        for label, score in ranked
        if score >= top_score - threshold
    ][:max_results]
    return {
        "top1":             kept[0][0],
        "above_threshold":  [l for l, _ in kept],
        "scores":           {l: round(s, 4) for l, s in kept},
    }


# ── Per-image extraction ───────────────────────────────────────────────────────

def extract_components(
    model,
    processor,
    image_path: Path,
    device: str,
    all_banks: dict[str, list[str]],
    all_text_feats: dict[str, torch.Tensor],
) -> dict:
    image = Image.open(image_path).convert("RGB")
    image_feat = encode_image(model, processor, image, device)

    components = {}
    for comp, labels in all_banks.items():
        result = score_and_threshold(image_feat, labels, all_text_feats[comp])
        components[comp] = result
        n = len(result["above_threshold"])
        print(f"  {comp:15s}  top1={result['top1']}  ({n} above threshold)")

    return components


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = default_device()
    print(f"Device: {device}\n")

    ci_vocab = load_ci_vocab()
    all_banks = {**ci_vocab, **FIXED_BANKS}
    print("Vocabulary sizes:")
    for k, v in all_banks.items():
        print(f"  {k}: {len(v)}")

    model, processor = load_clip(device)

    print("\nPre-computing text embeddings (done once for all images) …")
    all_text_feats: dict[str, torch.Tensor] = {}
    for comp, labels in all_banks.items():
        print(f"  encoding {comp} ({len(labels)} labels) …")
        all_text_feats[comp] = encode_texts(model, processor, labels, device)

    target_images = sorted(
        p for p in TARGET_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"\nFound {len(target_images)} target images.\n")

    results = {}
    for image_path in target_images:
        print(f"\n── {image_path.name} ──")
        results[image_path.name] = extract_components(
            model, processor, image_path, device, all_banks, all_text_feats
        )

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
