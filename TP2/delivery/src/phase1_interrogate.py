import json
from pathlib import Path

from PIL import Image

TARGET_DIR      = Path("statement/TP2-students/students/tp2-chosen")
PHASE1_CAPTIONS = Path("phase1_captions.json")
OUTPUT_FILE     = Path("phase1_warmstart.json")

CLIP_MODEL_NAME    = "ViT-L-14/openai"
CAPTION_MODEL_NAME = "blip-large"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def list_target_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Target directory not found: {path}")
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)

def load_subjects(path: Path) -> dict:
    if not path.exists():
        print(f"  ! {path} not found — warm start will use CI output only.")
        return {}
    data = json.loads(path.read_text())
    return {name: entry.get("subject", "") for name, entry in data.items()}

def build_warmstart(ci_prompt: str, subject: str) -> str:
    subject = (subject or "").strip().rstrip(".,")
    if not subject:
        return ci_prompt
    if ci_prompt.lower().lstrip().startswith(subject.lower()):
        return ci_prompt
    return f"{subject}, {ci_prompt}"

def main():
    from clip_interrogator import Config, Interrogator

    targets = list_target_images(TARGET_DIR)
    print(f"Found {len(targets)} target images.\n")

    subjects = load_subjects(PHASE1_CAPTIONS)

    print(f"Loading clip-interrogator (clip={CLIP_MODEL_NAME}, "
          f"caption={CAPTION_MODEL_NAME}) ...")
    ci = Interrogator(Config(
        clip_model_name=CLIP_MODEL_NAME,
        caption_model_name=CAPTION_MODEL_NAME,
    ))

    results: dict[str, dict] = {}
    for image_path in targets:
        image = Image.open(image_path).convert("RGB")
        subject = subjects.get(image_path.name, "")
        ci_prompt = ci.interrogate(image)
        warm = build_warmstart(ci_prompt, subject)
        results[image_path.name] = {
            "warm": warm,
            "ci": ci_prompt,
            "subject": subject,
        }
        print(f"\n── {image_path.name} ──")
        print(f"  CI:   {ci_prompt}")
        print(f"  warm: {warm}")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} warm-start prompts to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
