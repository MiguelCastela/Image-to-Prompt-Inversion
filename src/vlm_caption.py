"""
Phase 1 – Warm Start: VLM-based prompt initialisation.

Responsibilities are split by what each model is actually good at:
  - BLIP-2 (this script) produces the *content* description:
        base_caption – full-scene caption (unconditional)
        subject      – the main subject (short VQA answer)
  - CLIP-interrogator (phase1_clip.py -> phase1_clip.json) produces the
    *style* attributes (medium, artist_style, flavor, lighting, camera_angle)
    with confidence scores. CLIP does this far more reliably than BLIP-2's
    abstract VQA, so we do NOT duplicate it here.

The warm-start `assembled_prompt` is then built from BLIP's subject plus the
top-1 CLIP attribute for each style component. Run phase1_clip.py FIRST so
phase1_clip.json exists; otherwise the prompt falls back to subject only.

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
PHASE1_CLIP = Path("phase1_clip.json")   # style attributes (run phase1_clip.py first)

BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"

# blip2-opt-2.7b is ~7.8 GB in fp16 — just over an 8 GB card. Load it in 8-bit
# (same model, int8 weights, ~4 GB, near-lossless) so it fits the GPU. Set False
# to fall back to full-precision fp32 on CPU (slower but bit-exact).
LOAD_IN_8BIT = True

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Only ask BLIP-2 for the subject — the one structured field it answers reliably.
SUBJECT_QUESTION = "Question: What is the main subject of this image? Answer:"

# CLIP components pulled into the assembled prompt, in prompt order
# (strongest semantic tokens first: subject, then descriptive flavor, then style).
CLIP_ATTR_ORDER = ["flavor", "medium", "artist_style", "lighting", "camera_angle"]

# Meta-prefixes BLIP-2 tends to prepend; stripped so the subject is a clean token.
SUBJECT_PREFIXES = (
    "the main subject of this image is",
    "the main subject is",
    "the subject of this image is",
    "the subject is",
    "the main subject of the image is",
    "this image is of",
    "this image shows",
    "this image is",
    "it is",
    "a picture of",
    "an image of",
)


# ── Device ─────────────────────────────────────────────────────────────────────

def default_device() -> str:
    if torch.cuda.is_available() and LOAD_IN_8BIT:
        return "cuda"
    if torch.cuda.is_available() and not LOAD_IN_8BIT:
        # fp16 doesn't fit 8 GB; without 8-bit, run full precision on CPU.
        print("LOAD_IN_8BIT=False → using CPU (fp16 won't fit 8 GB GPU).")
        return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model ──────────────────────────────────────────────────────────────────────

def load_blip2(device: str):
    print(f"Loading {BLIP2_MODEL_ID} on {device} "
          f"({'8-bit' if device == 'cuda' and LOAD_IN_8BIT else 'full precision'}) ...")
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)

    if device == "cuda" and LOAD_IN_8BIT:
        from transformers import BitsAndBytesConfig
        model = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2_MODEL_ID,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2_MODEL_ID,
            torch_dtype=dtype,
        ).to(device)
    model.eval()
    return model, processor


def _prep_inputs(inputs, model):
    """Move processor outputs to the model device and match pixel dtype.

    8-bit keeps non-quantized weights (incl. the vision tower) in fp16, so the
    fp32 pixel_values from the processor must be cast to the model's dtype.
    """
    inputs = inputs.to(model.device)
    if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs


# ── Inference helpers ──────────────────────────────────────────────────────────

def generate_caption(model, processor, image: Image.Image, device: str) -> str:
    """Unconditional image caption (no question prompt)."""
    inputs = _prep_inputs(processor(images=image, return_tensors="pt"), model)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()


def answer_question(
    model, processor, image: Image.Image, question: str, device: str,
    max_new_tokens: int = 30,
) -> str:
    """Visual question answering with the BLIP-2 OPT decoder."""
    inputs = _prep_inputs(
        processor(images=image, text=question, return_tensors="pt"), model
    )
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    full = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    return clean_answer(full, question)


def clean_answer(raw: str, question: str) -> str:
    """Strip the echoed prompt framing and meta-prefixes from a VQA answer."""
    answer = raw.strip()
    # Drop everything up to and including a trailing "Answer:" the decoder echoed.
    if "Answer:" in answer:
        answer = answer.rsplit("Answer:", 1)[-1].strip()
    elif answer.startswith(question):
        answer = answer[len(question):].strip()
    # Keep only the first line / first sentence to avoid rambling hallucinations.
    answer = answer.split("\n")[0].strip()
    # Remove leading meta-prefixes ("The main subject of this image is ...").
    low = answer.lower()
    for prefix in SUBJECT_PREFIXES:
        if low.startswith(prefix):
            answer = answer[len(prefix):].lstrip(" :,").strip()
            break
    return answer.rstrip(".,").strip()


# ── CLIP style attributes ───────────────────────────────────────────────────────

def load_clip_top1(path: Path) -> dict:
    """Map image_name -> {component: top1_label} from phase1_clip.json, if present."""
    if not path.exists():
        print(f"  ! {path} not found — assembled_prompt will use subject only.")
        print(f"    (run phase1_clip.py first for style attributes)")
        return {}
    data = json.loads(path.read_text())
    return {
        name: {comp: entry[comp]["top1"] for comp in entry if "top1" in entry[comp]}
        for name, entry in data.items()
    }


# ── Per-image extraction ───────────────────────────────────────────────────────

def extract_components(
    model, processor, image_path: Path, device: str, clip_top1: dict
) -> dict:
    image = Image.open(image_path).convert("RGB")

    base_caption = generate_caption(model, processor, image, device)
    subject = answer_question(model, processor, image, SUBJECT_QUESTION, device)

    components = {
        "base_caption": base_caption,
        "subject": subject,
        # Style attributes sourced from CLIP (authoritative), not BLIP.
        "clip_style": clip_top1,
    }
    components["assembled_prompt"] = assemble_prompt(subject, clip_top1)
    return components


def assemble_prompt(subject: str, clip_top1: dict) -> str:
    """Warm-start prompt: BLIP subject + CLIP top-1 style attributes, in order."""
    parts = []
    if subject:
        parts.append(subject)
    for comp in CLIP_ATTR_ORDER:
        label = clip_top1.get(comp)
        if label:
            parts.append(label)
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

    clip_top1_all = load_clip_top1(PHASE1_CLIP)
    model, processor = load_blip2(device)

    results = {}
    for image_path in target_images:
        print(f"\n── {image_path.name} ──")
        clip_top1 = clip_top1_all.get(image_path.name, {})
        components = extract_components(
            model, processor, image_path, device, clip_top1
        )
        results[image_path.name] = components
        print(f"  base_caption: {components['base_caption']}")
        print(f"  subject: {components['subject']}")
        print(f"  assembled_prompt: {components['assembled_prompt']}")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved captions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
