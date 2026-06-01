"""
Phase 1 – Warm Start: VLM-based prompt initialisation using BLIP-2.

For each of the 6 target images, BLIP-2 is used to produce:
  - A base caption describing the full scene
  - Five structured components: subject, lighting, medium, camera angle, artist style

The five components are assembled into a single candidate prompt that serves as
the semantic baseline for Phase 2 (refinement).

Model: Salesforce/blip2-opt-2.7b  (downloaded automatically on first run)

Output: phase1_captions.json
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# ── Config ─────────────────────────────────────────────────────────────────────

TARGET_DIR = Path("statement/TP2-students/students/tp2-chosen")
OUTPUT_FILE = Path("phase1_captions.json")

BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Questions asked per image to extract each structured component.
# BLIP-2 OPT answers best with this "Question: ... Answer:" format.
COMPONENT_QUESTIONS = {
    "subject": (
        "Question: What is the main subject of this image? Answer:"
    ),
    "lighting": (
        "Question: How would you describe the lighting in this image "
        "(e.g. soft natural light, dramatic studio light, golden hour, neon)? Answer:"
    ),
    "medium": (
        "Question: What is the artistic medium of this image "
        "(e.g. digital art, oil painting, photograph, watercolour, 3D render)? Answer:"
    ),
    "camera_angle": (
        "Question: What is the camera angle or perspective "
        "(e.g. close-up, wide shot, bird's eye view, low angle, eye level)? Answer:"
    ),
    "artist_style": (
        "Question: What artistic style or artist does this image resemble "
        "(e.g. anime, photorealism, artgerm, Greg Rutkowski, concept art)? Answer:"
    ),
}


# ── Device ─────────────────────────────────────────────────────────────────────

def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model ──────────────────────────────────────────────────────────────────────

def load_blip2(device: str):
    print(f"Loading {BLIP2_MODEL_ID} on {device} ...")
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, processor


# ── Inference helpers ──────────────────────────────────────────────────────────

def generate_caption(model, processor, image: Image.Image, device: str) -> str:
    """Unconditional image caption (no question prompt)."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()


def answer_question(
    model, processor, image: Image.Image, question: str, device: str
) -> str:
    """Visual question answering with the BLIP-2 OPT decoder."""
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=60)
    full = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    # The decoder repeats the question prompt; strip it to keep only the answer.
    answer = full[len(question):].strip() if full.startswith(question) else full
    return answer


# ── Per-image extraction ───────────────────────────────────────────────────────

def extract_components(
    model, processor, image_path: Path, device: str
) -> dict:
    image = Image.open(image_path).convert("RGB")

    base_caption = generate_caption(model, processor, image, device)

    components = {"base_caption": base_caption}
    for component, question in COMPONENT_QUESTIONS.items():
        answer = answer_question(model, processor, image, question, device)
        components[component] = answer
        print(f"  {component}: {answer}")

    assembled = assemble_prompt(components)
    components["assembled_prompt"] = assembled
    return components


def assemble_prompt(components: dict) -> str:
    """
    Combine the five structured components into a single prompt string.
    Only non-empty fields are included.
    """
    parts = []

    if components.get("subject"):
        parts.append(components["subject"])
    if components.get("medium"):
        parts.append(components["medium"])
    if components.get("artist_style"):
        parts.append(components["artist_style"])
    if components.get("lighting"):
        parts.append(components["lighting"])
    if components.get("camera_angle"):
        parts.append(components["camera_angle"])

    return ", ".join(p.rstrip(".,") for p in parts if p)


# ── Main ───────────────────────────────────────────────────────────────────────

def list_target_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Target directory not found: {path}")
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    device = default_device()
    print(f"Device: {device}")

    target_images = list_target_images(TARGET_DIR)
    print(f"Found {len(target_images)} target images.\n")

    model, processor = load_blip2(device)

    results = {}
    for image_path in target_images:
        print(f"\n── {image_path.name} ──")
        components = extract_components(model, processor, image_path, device)
        results[image_path.name] = components
        print(f"  assembled_prompt: {components['assembled_prompt']}")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved captions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
