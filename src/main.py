import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import DiffusionPipeline
from IPython.display import display
from PIL import Image


# ── Paths ─────────────────────────────────────────────────────────────────────

TARGET_DIR = Path("statement/TP2-students/students/tp2-chosen")
OUTPUT_DIR = Path("statement/TP2-students/students/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ── Utilities: seeds, display, output folders ─────────────────────────────────

def seed_from_filename(path, fallback=2026):
    match = re.match(r"^(\d+)", Path(path).stem)
    return int(match.group(1)) if match else fallback


def safe_stem(path):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in Path(path).stem)


def load_image(path):
    return Image.open(path).convert("RGB")


def create_run_dir(base_dir=OUTPUT_DIR, identity="student_run"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"{timestamp}_{identity}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_csv(path, rows):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def show_images(paths, cols=3, title=None):
    paths = list(paths)
    if not paths:
        print("No images to show.")
        return
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    for ax in [ax for row in axes for ax in row]:
        ax.axis("off")
    for ax, path in zip([ax for row in axes for ax in row], paths):
        ax.imshow(load_image(path))
        ax.set_title(Path(path).name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def list_target_images(path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


# ── LCM config ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LCMConfig:
    model_id: str = "SimianLuo/LCM_Dreamshaper_v7"
    seed: int = 2026  # fallback only; target filenames define the real render seed
    num_inference_steps: int = 8
    guidance_scale: float = 8.0
    lcm_origin_steps: int = 50
    width: int = 768
    height: int = 768


config = LCMConfig()


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = default_device()
print("Using device:", device)


def load_lcm_pipeline(config):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe.to(device)
    return pipe


pipe = load_lcm_pipeline(config)


# ── Generate and save images ──────────────────────────────────────────────────

def render_prompt(prompt, seed, pipe=pipe, config=config):
    generator_device = "cpu" if device == "mps" else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        lcm_origin_steps=config.lcm_origin_steps,
        width=config.width,
        height=config.height,
        output_type="pil",
        generator=generator,
    ).images[0]
    return image


def render_prompt_for_target(prompt, target_path):
    seed = seed_from_filename(target_path, config.seed)
    return render_prompt(prompt, seed=seed)


def save_generated_image(image, run_dir, target_path, prompt_index=1):
    target_dir = Path(run_dir) / safe_stem(target_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"candidate_{prompt_index:03d}.png"
    image.save(path)
    return path


# ── Target images ─────────────────────────────────────────────────────────────

target_images = list_target_images(TARGET_DIR)
if not target_images:
    raise FileNotFoundError(
        f"No target images found in {TARGET_DIR}. "
        "Make sure the tp2-chosen folder is present."
    )

print("Target folder:", TARGET_DIR)
print("Output folder:", OUTPUT_DIR)
print("Number of targets:", len(target_images))
for path in target_images:
    print(Path(path).name, "-> seed", seed_from_filename(path))

show_images(target_images, cols=3, title="TP2 target images")


# ── Minimal example ───────────────────────────────────────────────────────────

run_dir = create_run_dir(identity="starter_example")

target_path = next(path for path in target_images if Path(path).name == "1159_25.png")
example_prompt = "orange juice"

generated = render_prompt_for_target(example_prompt, target_path)
saved_path = save_generated_image(generated, run_dir, target_path, prompt_index=1)

print("Target:", target_path)
print("Seed:", seed_from_filename(target_path))
print("Prompt:", example_prompt)
print("Saved generated image:", saved_path)

display(generated)


# ── Batch template for student prompts ────────────────────────────────────────
# Fill candidate_prompts with your recovered prompts (top-3 per target).
# Keys must match target filenames exactly.

candidate_prompts = {
    "1159_25.png": [
        "Orange Juice",
    ],
}

run_dir = create_run_dir(identity="student_prompts")
rows = []

target_by_name = {Path(path).name: path for path in target_images}
for image_name, prompts in candidate_prompts.items():
    target_path = target_by_name.get(image_name)
    if target_path is None:
        print("Skipping unknown target:", image_name)
        continue
    seed = seed_from_filename(target_path, config.seed)
    for index, prompt in enumerate(prompts, start=1):
        generated = render_prompt(prompt, seed=seed)
        image_path = save_generated_image(generated, run_dir, target_path, prompt_index=index)
        rows.append({
            "target": str(target_path),
            "target_name": image_name,
            "render_seed": seed,
            "candidate_index": index,
            "prompt": prompt,
            "render": str(image_path),
        })

write_csv(run_dir / "generated_prompts.csv", rows)
(run_dir / "generated_prompts.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
print("Saved run to:", run_dir)
print("Generated", len(rows), "image(s)")


# ── View generated outputs ────────────────────────────────────────────────────

generated_paths = [row["render"] for row in rows]
show_images(generated_paths, cols=3, title="Generated candidates")
