"""
Phase 1 – Warm Start (CLIP branch): zero-shot component extraction.

Unlike BLIP-2, CLIP cannot generate text. Instead it scores a predefined list
of candidate labels against the image embedding and returns the best matches.
This gives a structured, DreamShaper-style decomposition of each target image.

For each target image this file extracts:
  subject, lighting, medium, camera angle, artist style

The top-1 label per component is assembled into a candidate prompt.
Top-3 per component are also saved so Phase 2 can explore combinations.

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

TOP_K = 3  # how many candidates to keep per component


# ── Candidate label banks ──────────────────────────────────────────────────────
# These are phrased as image descriptions rather than bare nouns so CLIP's
# text encoder gets proper context.

CANDIDATES = {
    "medium": [
        "digital art",
        "digital painting",
        "oil painting",
        "watercolour painting",
        "acrylic painting",
        "pencil sketch",
        "charcoal drawing",
        "photograph",
        "photorealistic render",
        "3D CGI render",
        "anime illustration",
        "manga artwork",
        "concept art",
        "fantasy illustration",
        "vector illustration",
        "pixel art",
    ],
    "lighting": [
        "soft natural lighting",
        "golden hour lighting",
        "studio lighting",
        "dramatic cinematic lighting",
        "volumetric lighting",
        "rim lighting",
        "neon lighting",
        "candlelight",
        "backlit",
        "ambient occlusion lighting",
        "harsh directional lighting",
        "moody atmospheric lighting",
        "bright even lighting",
        "dark dramatic shadows",
        "iridescent lighting",
    ],
    "camera_angle": [
        "extreme close-up shot",
        "close-up portrait shot",
        "medium shot",
        "full body shot",
        "wide angle shot",
        "bird's eye view",
        "low angle shot",
        "eye level shot",
        "overhead shot",
        "dutch angle shot",
        "macro photography shot",
        "establishing wide shot",
    ],
    "artist_style": [
        "by artgerm",
        "by Greg Rutkowski",
        "by WLOP",
        "by Alphonse Mucha",
        "by Makoto Shinkai",
        "by Frank Frazetta",
        "by Ross Tran",
        "by Charlie Bowater",
        "by Ilya Kuvshinov",
        "Studio Ghibli style",
        "anime style",
        "dark fantasy art style",
        "hyperrealistic style",
        "sci-fi concept art style",
        "painterly impressionist style",
        "sharp detailed fantasy art",
    ],
    "subject": [
        "a portrait of a person",
        "a fantasy character in armour",
        "an anime character",
        "a creature or monster",
        "a landscape scene",
        "a cityscape",
        "a food or drink item",
        "an animal",
        "a mythical creature",
        "an astronaut or space explorer",
        "a natural environment",
        "an abstract composition",
        "a still life",
        "a dragon",
        "a robot or mechanical figure",
    ],
}


# ── Device ─────────────────────────────────────────────────────────────────────

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model ──────────────────────────────────────────────────────────────────────

def load_clip(device: str):
    print(f"Loading {CLIP_MODEL_ID} on {device} ...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    model.eval()
    return model, processor


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_candidates(
    model, processor, image: Image.Image, candidates: list[str], device: str
) -> list[tuple[str, float]]:
    """
    Return candidates sorted by CLIP cosine similarity to the image (descending).
    """
    inputs = processor(
        text=candidates,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    scores = (image_features @ text_features.T).squeeze(0)
    ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked


# ── Per-image extraction ───────────────────────────────────────────────────────

def extract_components(
    model, processor, image_path: Path, device: str
) -> dict:
    image = Image.open(image_path).convert("RGB")

    components = {}
    for component, candidates in CANDIDATES.items():
        ranked = score_candidates(model, processor, image, candidates, device)
        top_k = ranked[:TOP_K]
        components[component] = {
            "top1": top_k[0][0],
            "top3": [label for label, _ in top_k],
            "scores": {label: round(score, 4) for label, score in top_k},
        }
        print(f"  {component}: {top_k[0][0]}  (score {top_k[0][1]:.4f})")

    assembled = assemble_prompt(components)
    components["assembled_prompt"] = assembled
    return components


def assemble_prompt(components: dict) -> str:
    """
    Build a DreamShaper-style prompt from top-1 labels.
    Order: subject, medium, artist style, lighting, camera angle.
    """
    order = ["subject", "medium", "artist_style", "lighting", "camera_angle"]
    parts = [components[k]["top1"] for k in order if k in components]
    return ", ".join(parts)


# ── Main ───────────────────────────────────────────────────────────────────────

def list_target_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Target directory not found: {path}")
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    device = default_device()
    print(f"Device: {device}\n")

    target_images = list_target_images(TARGET_DIR)
    print(f"Found {len(target_images)} target images.\n")

    model, processor = load_clip(device)

    results = {}
    for image_path in target_images:
        print(f"\n── {image_path.name} ──")
        components = extract_components(model, processor, image_path, device)
        results[image_path.name] = components
        print(f"  assembled_prompt: {components['assembled_prompt']}")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
