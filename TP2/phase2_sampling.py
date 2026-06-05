"""
Phase 2 – Prompt Sampling: LLM-based prompt expansion.

Merges Phase 1 CLIP + BLIP-2 outputs into a rich baseline per image, then
calls the Claude API to generate N_VARIATIONS diverse DreamShaper-style
prompt variants for each target image.

Input:  phase1_clip.json, phase1_captions.json
Output: phase2_candidates.json  (image_name -> list[str])
"""

import json
import re
from pathlib import Path

import anthropic


PHASE1_CLIP = Path("phase1_clip.json")
PHASE1_CAPTIONS = Path("phase1_captions.json")
OUTPUT_FILE = Path("phase2_candidates.json")

N_VARIATIONS = 30
MODEL_ID = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are an expert prompt engineer for Stable Diffusion image generators, \
specifically DreamShaper v7 running with LCM sampling (8 inference steps, \
guidance scale 8.0, fixed random seed).

Your task: given a description of a target image, generate exactly {n} diverse \
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
- Try different orderings — earlier tokens have stronger CLIP influence
- Use comma-separated format; no full sentences

Output EXACTLY {n} prompts, numbered "1." through "{n}.". No preamble or commentary.\
"""


CLIP_COMPONENTS = ["medium", "artist_style", "flavor", "lighting", "camera_angle"]


def build_baseline(clip_data: dict, caption_data: dict, image_name: str) -> dict:
    return {
        "clip": clip_data.get(image_name, {}),
        "blip_caption": caption_data.get(image_name, {}).get("base_caption", ""),
        "blip_subject": caption_data.get(image_name, {}).get("subject", ""),
    }


def _fmt_clip_component(comp: dict) -> str:
    return ", ".join(
        f"{label} ({score:.3f})"
        for label, score in comp.get("scores", {}).items()
    )


def parse_numbered_list(text: str, n: int) -> list[str]:
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    prompts = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            prompts.append(cleaned)
    return prompts[:n]


def generate_variations(
    client: anthropic.Anthropic, baseline: dict, n: int = N_VARIATIONS
) -> list[str]:
    clip = baseline["clip"]
    clip_lines = "\n".join(
        f"  {comp}: {_fmt_clip_component(clip[comp])}"
        for comp in CLIP_COMPONENTS
        if comp in clip
    )
    blip_lines = f"  {baseline['blip_caption']}"
    if baseline["blip_subject"]:
        blip_lines += f"\n  subject answer: {baseline['blip_subject']}"

    user_msg = (
        f"CLIP (label: cosine similarity score):\n{clip_lines}\n\n"
        f"BLIP-2 (free-text caption):\n{blip_lines}\n\n"
        f"Generate {n} diverse prompt variations for this image."
    )

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=4096,
        system=SYSTEM_PROMPT.format(n=n),
        messages=[{"role": "user", "content": user_msg}],
    )

    return parse_numbered_list(response.content[0].text, n)


def main():
    clip_data = json.loads(PHASE1_CLIP.read_text())
    caption_data = json.loads(PHASE1_CAPTIONS.read_text())
    client = anthropic.Anthropic()

    results: dict[str, list[str]] = {}

    for image_name in clip_data:
        print(f"\n── {image_name} ──")
        baseline = build_baseline(clip_data, caption_data, image_name)
        print(f"  Caption:  {baseline['blip_caption']}")

        variations = generate_variations(client, baseline)
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
