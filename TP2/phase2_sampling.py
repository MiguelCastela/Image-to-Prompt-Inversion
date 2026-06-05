"""
Phase 2 – Prompt Sampling: VLM-based prompt expansion.

Each target image is shown directly to the VLM together with the Phase 1
CLIP scores and BLIP-2 caption as additional context. The model generates
N_VARIATIONS diverse DreamShaper-style prompt candidates per image.

Backends (switch via BACKEND constant):
  qwen   – Qwen/Qwen2.5-VL-7B-Instruct-AWQ  (local, 4-bit AWQ, runs on RTX)
  claude – claude-opus-4-8                    (Anthropic API, for comparison)

Input:  phase1_clip.json, phase1_captions.json, target images in TARGET_DIR
Output: phase2_candidates.json  (image_name -> list[str])
"""

import base64
import json
import re
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

PHASE1_CLIP     = Path("phase1_clip.json")
PHASE1_CAPTIONS = Path("phase1_captions.json")
TARGET_DIR      = Path("statement/TP2-students/students/tp2-chosen")
OUTPUT_FILE     = Path("phase2_candidates.json")

BACKEND         = "qwen"    # "qwen" | "claude"
QWEN_MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
CLAUDE_MODEL_ID = "claude-opus-4-8"
N_VARIATIONS    = 30

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CLIP_COMPONENTS  = ["medium", "artist_style", "flavor", "lighting", "camera_angle"]


# ── Shared prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert prompt engineer for Stable Diffusion image generators, \
specifically DreamShaper v7 running with LCM sampling (8 inference steps, \
guidance scale 8.0, fixed random seed).

Your task: given a target image and its analysis, generate exactly {n} diverse \
prompt variations. These will be rendered with a fixed random seed and compared \
pixel-by-pixel against the original. Closer pixels = better score.

Rules for high-scoring DreamShaper v7 prompts:
- Put the strongest semantic tokens FIRST (subject, then style descriptors)
- Include quality boosters: masterpiece, best quality, highly detailed, sharp focus, 8k
- Vary artist references across variations: artgerm, Greg Rutkowski, WLOP, \
Alphonse Mucha, Makoto Shinkai, Ross Tran, Charlie Bowater, Ilya Kuvshinov
- Vary medium: digital painting, oil painting, concept art, anime, 3D render, \
photorealistic, watercolour, fantasy illustration
- Vary lighting: volumetric lighting, studio lighting, dramatic cinematic lighting, \
golden hour, rim lighting, ambient occlusion, neon lighting
- Vary composition: close-up portrait, medium shot, full body, wide angle, \
bokeh, depth of field, rule of thirds
- Try different token orderings — earlier tokens have stronger CLIP influence
- Use comma-separated format; no full sentences

Output EXACTLY {n} prompts, numbered "1." through "{n}.". No preamble or commentary.\
"""


# ── Context helpers ───────────────────────────────────────────────────────────

def _fmt_component(comp: dict) -> str:
    return ", ".join(
        f"{label} ({score:.3f})"
        for label, score in comp.get("scores", {}).items()
    )


def build_text_context(clip: dict, blip_caption: str, blip_subject: str) -> str:
    clip_lines = "\n".join(
        f"  {comp}: {_fmt_component(clip[comp])}"
        for comp in CLIP_COMPONENTS
        if comp in clip
    )
    blip_lines = f"  {blip_caption}"
    if blip_subject:
        blip_lines += f"\n  subject: {blip_subject}"
    return (
        f"CLIP (label: cosine similarity):\n{clip_lines}\n\n"
        f"BLIP-2 (free-text caption):\n{blip_lines}"
    )


def parse_numbered_list(text: str, n: int) -> list[str]:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    prompts = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            prompts.append(cleaned)
    return prompts[:n]


# ── Qwen backend ──────────────────────────────────────────────────────────────

def load_qwen(model_id: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"Loading {model_id} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print("Model loaded.\n")
    return model, processor


def generate_qwen(
    model, processor, image_path: Path, context: str, n: int
) -> list[str]:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {
                    "type": "text",
                    "text": f"{context}\n\nGenerate {n} diverse prompt variations for this image.",
                },
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return parse_numbered_list(output, n)


# ── Claude backend ────────────────────────────────────────────────────────────

def load_claude():
    import anthropic
    return anthropic.Anthropic()


def generate_claude(
    client, image_path: Path, context: str, n: int
) -> list[str]:
    ext = image_path.suffix.lower().lstrip(".")
    media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
    img_data = base64.standard_b64encode(image_path.read_bytes()).decode()

    response = client.messages.create(
        model=CLAUDE_MODEL_ID,
        max_tokens=4096,
        system=SYSTEM_PROMPT.format(n=n),
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data,
                    },
                },
                {
                    "type": "text",
                    "text": f"{context}\n\nGenerate {n} diverse prompt variations for this image.",
                },
            ],
        }],
    )
    return parse_numbered_list(response.content[0].text, n)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    clip_data    = json.loads(PHASE1_CLIP.read_text())
    caption_data = json.loads(PHASE1_CAPTIONS.read_text())

    target_by_name = {
        p.name: p
        for p in TARGET_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }

    if BACKEND == "qwen":
        model, processor = load_qwen(QWEN_MODEL_ID)
        client = None
    else:
        model = processor = None
        client = load_claude()

    results: dict[str, list[str]] = {}

    for image_name in clip_data:
        target_path = target_by_name.get(image_name)
        if target_path is None:
            print(f"Skipping {image_name}: target image not found")
            continue

        print(f"\n── {image_name} ──")
        context = build_text_context(
            clip_data[image_name],
            caption_data.get(image_name, {}).get("base_caption", ""),
            caption_data.get(image_name, {}).get("subject", ""),
        )

        if BACKEND == "qwen":
            variations = generate_qwen(model, processor, target_path, context, N_VARIATIONS)
        else:
            variations = generate_claude(client, target_path, context, N_VARIATIONS)

        results[image_name] = variations
        print(f"  Generated {len(variations)} variations. First 3:")
        for i, v in enumerate(variations[:3], 1):
            print(f"    {i}. {v}")
        if len(variations) > 3:
            print(f"    ... ({len(variations) - 3} more)")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    total = sum(len(v) for v in results.values())
    print(f"\nSaved {total} candidates across {len(results)} images to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
