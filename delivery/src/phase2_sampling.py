import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path

WARMSTART_FILE  = Path("phase1_warmstart.json")
PHASE1_CAPTIONS = Path("phase1_captions.json")
TARGET_DIR      = Path("statement/TP2-students/students/tp2-chosen")
OUTPUT_FILE     = Path("phase2_candidates.json")
FULL_OUTPUT_FILE = Path("phase2_full.json")

BACKEND         = "qwen"
QWEN_MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
CLAUDE_MODEL_ID = "claude-opus-4-8"
N_VARIATIONS    = 30
MAX_NEW_TOKENS  = 1024

DO_SAMPLE     = True
TEMPERATURE   = 0.8
TOP_P         = 0.95
ROUND_SIZE    = 10
MAX_ROUNDS    = 6
GEN_SEED      = 1234

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SYSTEM_PROMPT = """\
You are a prompt engineer for the DreamShaper v7 Stable Diffusion model (LCM \
sampling, 8 inference steps, guidance scale 8.0, fixed random seed).

Given a TARGET image and its analysis, write exactly {n} DIFFERENT prompts that, \
when rendered, reproduce the target as closely as possible. Each render is \
compared against the target, so every prompt must depict the SAME target.

Rules:
- Describe ONLY what is visible in the target: subject, colours, composition, \
background, style, lighting.
- Put the SUBJECT first — it is the most important token.
- Treat the BLIP-2 text as the SUBJECT (what the thing is) and the \
clip-interrogator text as STYLE HINTS (how it looks).
- The {n} prompts MUST be genuinely different from each other: do NOT copy the \
style hints verbatim. Recombine them, rephrase with synonyms, reorder tokens, \
add or drop descriptive details, and vary emphasis across the {n} prompts — \
while still depicting the same target.
- Do NOT invent artists, people, or content that is not implied by the target.
- Use short comma-separated tags, not full sentences. Keep each under 30 words.

Output EXACTLY {n} prompts, numbered "1." to "{n}.", one per line. No preamble, \
no commentary, no markdown, no blank lines.\
"""

def build_text_context(subject: str, ci_style: str, blip_caption: str) -> str:
    lines = []
    if subject:
        lines.append(f"SUBJECT (BLIP-2, what the thing is): {subject}")
    if blip_caption:
        lines.append(f"BLIP-2 caption: {blip_caption}")
    if ci_style:
        lines.append(f"STYLE / medium (clip-interrogator): {ci_style}")
    return "\n".join(lines)

_LABEL_RE = re.compile(
    r"^(?:artist_style|camera_angle|composition|lighting|medium|subject|flavor|style|artist)\s*:\s*",
    re.IGNORECASE,
)

def clean_prompt(prompt: str) -> str:
    tokens = (_LABEL_RE.sub("", t.strip()) for t in prompt.split(","))
    return ", ".join(t for t in (tok.strip() for tok in tokens) if t)

def dedupe(prompts: list[str], n: int) -> list[str]:
    out, seen = [], set()
    for p in prompts:
        key = p.lower()
        if p and key not in seen:
            seen.add(key)
            out.append(p)
    return out[:n]

_NUM_RE = re.compile(
    r"^\s*\*{0,2}\s*(?:prompt\s*)?#?\s*\d+\s*[.):]\s*\*{0,2}\s*(.*\S)\s*$",
    re.IGNORECASE,
)

def parse_numbered_list(text: str, n: int) -> list[str]:
    prompts = []
    for line in text.split("\n"):
        m = _NUM_RE.match(line)
        if not m:
            continue
        body = m.group(1).strip().strip("*`").strip()
        cleaned = clean_prompt(body)
        if cleaned:
            prompts.append(cleaned)
    return dedupe(prompts, n)

def load_qwen(model_id: str):
    import torch
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )
    from transformers.quantizers.quantizer_awq import AwqQuantizer

    AwqQuantizer.update_dtype = lambda self, dtype: dtype

    import huggingface_hub as _hf
    from transformers.utils import hub as _th
    for _name in ("create_repo", "list_repo_tree", "has_file", "cached_file",
                  "hf_hub_download", "snapshot_download"):
        if not hasattr(_th, _name) and hasattr(_hf, _name):
            setattr(_th, _name, getattr(_hf, _name))

    print(f"Loading {model_id} ...")

    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "quantization_config", None) is not None:
        config.quantization_config["backend"] = "gemm_triton"
        config.quantization_config["modules_to_not_convert"] = ["model.visual"]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print("Model loaded.\n")
    return model, processor

def _round_user_text(context: str, need: int, history: list[str]) -> str:
    text = f"{context}\n\nGenerate {need} different prompts for this target."
    if history:
        avoid = "\n".join(f"- {p}" for p in history)
        text += (
            f"\n\nYou have ALREADY generated the prompts below. Generate {need} "
            f"NEW prompts that are clearly DIFFERENT from every one of these "
            f"(different wording, emphasis, and detail):\n{avoid}"
        )
    return text

def generate_qwen(
    model, processor, image_path: Path, context: str, n: int
) -> tuple[list[str], str]:
    import torch
    from qwen_vl_utils import process_vision_info

    torch.manual_seed(GEN_SEED)
    collected: list[str] = []
    raw_rounds: list[str] = []

    for _ in range(MAX_ROUNDS):
        if len(collected) >= n:
            break
        need = min(ROUND_SIZE, n - len(collected))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(n=need)},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path.resolve().as_uri()},
                    {"type": "text", "text": _round_user_text(context, need, collected)},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        output = processor.decode(
            generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        raw_rounds.append(output)

        before = len(collected)

        collected = dedupe(collected + parse_numbered_list(output, n), n)
        if len(collected) == before:
            break

    return collected, "\n--- round ---\n".join(raw_rounds)

def load_claude():
    import anthropic
    return anthropic.Anthropic()

def generate_claude(
    client, image_path: Path, context: str, n: int
) -> tuple[list[str], str]:
    ext = image_path.suffix.lower().lstrip(".")
    media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
    img_data = base64.standard_b64encode(image_path.read_bytes()).decode()

    response = client.messages.create(
        model=CLAUDE_MODEL_ID,
        max_tokens=4096,
        temperature=TEMPERATURE,
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
                    "text": f"{context}\n\nGenerate {n} different prompts for this target.",
                },
            ],
        }],
    )
    raw = response.content[0].text
    return parse_numbered_list(raw, n), raw

def main():
    warmstarts   = json.loads(WARMSTART_FILE.read_text())
    caption_data = json.loads(PHASE1_CAPTIONS.read_text())

    target_by_name = {
        p.name: p
        for p in TARGET_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }

    if BACKEND == "qwen":
        model, processor = load_qwen(QWEN_MODEL_ID)
        client = None

        from transformers import set_seed
        set_seed(GEN_SEED)
        print(f"Sampling: do_sample={DO_SAMPLE} temperature={TEMPERATURE} "
              f"top_p={TOP_P} seed={GEN_SEED}")
    else:
        model = processor = None
        client = load_claude()

    model_id = QWEN_MODEL_ID if BACKEND == "qwen" else CLAUDE_MODEL_ID
    system_prompt = SYSTEM_PROMPT.format(n=N_VARIATIONS)

    results: dict[str, list[str]] = {}
    full: dict[str, dict] = {}

    for image_name, ws in warmstarts.items():
        target_path = target_by_name.get(image_name)
        if target_path is None:
            print(f"Skipping {image_name}: target image not found")
            continue

        print(f"\n── {image_name} ──")
        cap = caption_data.get(image_name, {})

        if isinstance(ws, dict):
            subject  = ws.get("subject") or cap.get("subject", "")
            ci_style = ws.get("ci", "")
        else:
            subject  = cap.get("subject", "")
            ci_style = ws
        context = build_text_context(
            subject, ci_style, cap.get("base_caption", ""),
        )

        if BACKEND == "qwen":
            variations, raw_output = generate_qwen(
                model, processor, target_path, context, N_VARIATIONS
            )
        else:
            variations, raw_output = generate_claude(
                client, target_path, context, N_VARIATIONS
            )

        results[image_name] = variations
        full[image_name] = {
            "image_path": str(target_path),
            "context": context,
            "raw_output": raw_output,
            "variations": variations,
            "n_parsed": len(variations),
            "n_requested": N_VARIATIONS,
        }
        print(f"  Generated {len(variations)} variations. First 3:")
        for i, v in enumerate(variations[:3], 1):
            print(f"    {i}. {v}")
        if len(variations) > 3:
            print(f"    ... ({len(variations) - 3} more)")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    total = sum(len(v) for v in results.values())
    print(f"\nSaved {total} candidates across {len(results)} images to {OUTPUT_FILE}")

    full_record = {
        "metadata": {
            "backend": BACKEND,
            "model_id": model_id,
            "n_variations": N_VARIATIONS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE if DO_SAMPLE else None,
            "top_p": TOP_P if DO_SAMPLE else None,
            "gen_seed": GEN_SEED,
            "round_size": ROUND_SIZE,
            "max_rounds": MAX_ROUNDS,
            "sampling_mode": "round-based with prompt history (avoid-list conditioning)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system_prompt": system_prompt,
            "warmstart_source": str(WARMSTART_FILE),
            "n_images": len(full),
            "n_total_candidates": total,
        },
        "images": full,
    }
    FULL_OUTPUT_FILE.write_text(
        json.dumps(full_record, indent=2, ensure_ascii=False)
    )
    print(f"Saved full report record to {FULL_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
