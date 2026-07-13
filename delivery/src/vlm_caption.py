import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

TARGET_DIR = Path("statement/TP2-students/students/tp2-chosen")
OUTPUT_FILE = Path("phase1_captions.json")
PHASE1_CLIP = Path("phase1_clip.json")

BLIP2_MODEL_ID = "Salesforce/blip2-opt-2.7b"

LOAD_IN_8BIT = True

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SUBJECT_QUESTION = "Question: What is the main subject of this image? Answer:"

CLIP_ATTR_ORDER = ["flavor", "medium", "artist_style", "lighting", "camera_angle"]

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

def default_device() -> str:
    if torch.cuda.is_available() and LOAD_IN_8BIT:
        return "cuda"
    if torch.cuda.is_available() and not LOAD_IN_8BIT:

        print("LOAD_IN_8BIT=False → using CPU (fp16 won't fit 8 GB GPU).")
        return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

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
    inputs = inputs.to(model.device)
    if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    return inputs

def generate_caption(model, processor, image: Image.Image, device: str) -> str:
    inputs = _prep_inputs(processor(images=image, return_tensors="pt"), model)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()

def answer_question(
    model, processor, image: Image.Image, question: str, device: str,
    max_new_tokens: int = 30,
) -> str:
    inputs = _prep_inputs(
        processor(images=image, text=question, return_tensors="pt"), model
    )
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    full = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    return clean_answer(full, question)

def clean_answer(raw: str, question: str) -> str:
    answer = raw.strip()

    if "Answer:" in answer:
        answer = answer.rsplit("Answer:", 1)[-1].strip()
    elif answer.startswith(question):
        answer = answer[len(question):].strip()

    answer = answer.split("\n")[0].strip()

    low = answer.lower()
    for prefix in SUBJECT_PREFIXES:
        if low.startswith(prefix):
            answer = answer[len(prefix):].lstrip(" :,").strip()
            break
    return answer.rstrip(".,").strip()

def load_clip_top1(path: Path) -> dict:
    if not path.exists():
        print(f"  ! {path} not found — assembled_prompt will use subject only.")
        print(f"    (run phase1_clip.py first for style attributes)")
        return {}
    data = json.loads(path.read_text())
    return {
        name: {comp: entry[comp]["top1"] for comp in entry if "top1" in entry[comp]}
        for name, entry in data.items()
    }

def extract_components(
    model, processor, image_path: Path, device: str, clip_top1: dict
) -> dict:
    image = Image.open(image_path).convert("RGB")

    base_caption = generate_caption(model, processor, image, device)
    subject = answer_question(model, processor, image, SUBJECT_QUESTION, device)

    components = {
        "base_caption": base_caption,
        "subject": subject,

        "clip_style": clip_top1,
    }
    components["assembled_prompt"] = assemble_prompt(subject, clip_top1)
    return components

def assemble_prompt(subject: str, clip_top1: dict) -> str:
    parts = []
    if subject:
        parts.append(subject)
    for comp in CLIP_ATTR_ORDER:
        label = clip_top1.get(comp)
        if label:
            parts.append(label)
    return ", ".join(p.rstrip(".,") for p in parts if p)

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
