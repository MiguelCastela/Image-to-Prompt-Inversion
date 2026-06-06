"""
Phase 2 – Prompt Sampling: open-loop VLM prompt expansion (baseline).

Each target image is shown directly to the VLM together with the warm-start
context: the BLIP-2 subject/caption and the clip-interrogator style string. The
model generates N_VARIATIONS diverse DreamShaper-style prompt candidates per
image in one pass. This is the open-loop baseline; Phase 4 adds the closed-loop
render/score feedback on top of the same warm start.

Backends (switch via BACKEND constant):
  qwen   – Qwen/Qwen2.5-VL-7B-Instruct-AWQ  (local, 4-bit AWQ, runs on RTX)
  claude – claude-opus-4-8                    (Anthropic API, for comparison)

Input:  phase1_warmstart.json (from phase1_interrogate.py), phase1_captions.json,
        target images in TARGET_DIR
Output: phase2_candidates.json  (image_name -> list[str])
"""

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

WARMSTART_FILE  = Path("phase1_warmstart.json")    # from phase1_interrogate.py
PHASE1_CAPTIONS = Path("phase1_captions.json")
TARGET_DIR      = Path("statement/TP2-students/students/tp2-chosen")
OUTPUT_FILE     = Path("phase2_candidates.json")   # pipeline input for phase 3
FULL_OUTPUT_FILE = Path("phase2_full.json")        # rich record for the report

BACKEND         = "qwen"    # "qwen" | "claude"
QWEN_MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
CLAUDE_MODEL_ID = "claude-opus-4-8"
N_VARIATIONS    = 30
MAX_NEW_TOKENS  = 4096

# Sampling: greedy decoding collapses the N variations toward near-duplicates,
# so sample with a moderate temperature for diversity. Reproducibility is kept
# by fixing GEN_SEED before generation (logged in phase2_full.json), not by
# greedy decoding.
DO_SAMPLE   = True
TEMPERATURE = 0.7
TOP_P       = 0.9
GEN_SEED    = 1234

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# ── Shared prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a prompt engineer for the DreamShaper v7 Stable Diffusion model (LCM \
sampling, 8 inference steps, guidance scale 8.0, fixed random seed).

Given a TARGET image and its analysis, write exactly {n} diverse prompts that, \
when rendered, reproduce the target as closely as possible. Each render is \
compared against the target, so every prompt must depict the SAME target.

Rules:
- Describe ONLY what is visible in the target: subject, colours, composition, \
background, style, lighting.
- Put the SUBJECT first — it is the most important token.
- Treat the BLIP-2 text as the SUBJECT (what the thing is) and the \
clip-interrogator text as the STYLE / medium (how it looks).
- Do NOT invent artists, people, or content that is not implied by the target.
- Vary the wording, token ordering, and emphasis across the {n} prompts so they \
explore the space — but every prompt must still depict the same target.
- Use short comma-separated tags, not full sentences. Keep each prompt under 30 words.

Output EXACTLY {n} prompts, numbered "1." through "{n}.". No preamble or commentary.\
"""


# ── Context helpers ───────────────────────────────────────────────────────────

def build_text_context(subject: str, ci_style: str, blip_caption: str) -> str:
    """Context from the warm start: BLIP-2 subject/caption + clip-interrogator
    style string, labelled so the VLM keeps subject and style distinct."""
    lines = []
    if subject:
        lines.append(f"SUBJECT (BLIP-2, what the thing is): {subject}")
    if blip_caption:
        lines.append(f"BLIP-2 caption: {blip_caption}")
    if ci_style:
        lines.append(f"STYLE / medium (clip-interrogator): {ci_style}")
    return "\n".join(lines)


# The VLM sometimes echoes the analysis field labels into a prompt, e.g.
# "..., artist_style: Greg Rutkowski, medium: digital painting, lighting: ...".
# Strip those leading "label:" prefixes per comma-token so the prompt keeps only
# the value ("Greg Rutkowski", "digital painting", ...) as clean SD tokens.
_LABEL_RE = re.compile(
    r"^(?:artist_style|camera_angle|composition|lighting|medium|subject|flavor|style|artist)\s*:\s*",
    re.IGNORECASE,
)


def clean_prompt(prompt: str) -> str:
    tokens = (_LABEL_RE.sub("", t.strip()) for t in prompt.split(","))
    return ", ".join(t for t in (tok.strip() for tok in tokens) if t)


def parse_numbered_list(text: str, n: int) -> list[str]:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    prompts = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            prompts.append(clean_prompt(cleaned))
    return prompts[:n]


# ── Qwen backend ──────────────────────────────────────────────────────────────

def load_qwen(model_id: str):
    import torch
    from transformers import (
        AutoConfig,
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )
    from transformers.quantizers.quantizer_awq import AwqQuantizer

    # AwqQuantizer.update_dtype force-downcasts bf16 -> fp16 as a guard for the
    # CUDA AWQ kernels. We use the Triton kernel, which DOES support bf16 and
    # emits bf16 — the downcast leaves lm_head in fp16 and causes a
    # "BFloat16 != Half" mismatch. Keep the requested dtype as-is.
    AwqQuantizer.update_dtype = lambda self, dtype: dtype

    print(f"Loading {model_id} ...")
    # Two fixes to the model's AWQ config for transformers 5.x + gptqmodel:
    #
    # 1. modules_to_not_convert: the checkpoint leaves the vision tower in fp16,
    #    but its exclusion pattern is "visual" while the layers are actually named
    #    "model.visual.*". transformers' matcher anchors at the start (re.match),
    #    so "visual" never matches and it tries to AWQ-quantize the vision MLP
    #    (intermediate_size 3420, not divisible by group_size 128 → crash).
    #    Use "model.visual" so the whole vision tower stays fp16.
    #
    # 2. backend: default "auto" picks Marlin (needs out_features %% 64); the plain
    #    "gemm" kernel JIT-builds a CUDA C++ extension needing nvcc >= 12.8 for
    #    Blackwell (this box has nvcc 12.0). "gemm_triton" handles the language
    #    layers and compiles via Triton (no nvcc).
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "quantization_config", None) is not None:
        config.quantization_config["backend"] = "gemm_triton"
        config.quantization_config["modules_to_not_convert"] = ["model.visual"]

    # Use bfloat16 (Qwen2.5-VL's native dtype). The Triton AWQ kernel dequantizes
    # to bf16, so the non-quantized layers (lm_head, norms, vision) must also be
    # bf16 — forcing fp16 causes a "BFloat16 != Half" mismatch at lm_head.
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


def generate_qwen(
    model, processor, image_path: Path, context: str, n: int
) -> tuple[list[str], str]:
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

    gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE}
    if DO_SAMPLE:
        gen_kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return parse_numbered_list(output, n), output


# ── Claude backend ────────────────────────────────────────────────────────────

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
    raw = response.content[0].text
    return parse_numbered_list(raw, n), raw


# ── Main ──────────────────────────────────────────────────────────────────────

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
        # Fix the RNG so temperature sampling is reproducible for the report.
        from transformers import set_seed
        set_seed(GEN_SEED)
        print(f"Sampling: do_sample={DO_SAMPLE} temperature={TEMPERATURE} "
              f"top_p={TOP_P} seed={GEN_SEED}")
    else:
        model = processor = None
        client = load_claude()

    model_id = QWEN_MODEL_ID if BACKEND == "qwen" else CLAUDE_MODEL_ID
    system_prompt = SYSTEM_PROMPT.format(n=N_VARIATIONS)

    results: dict[str, list[str]] = {}            # phase-3 input (name -> prompts)
    full: dict[str, dict] = {}                     # rich per-image record for report

    for image_name, ws in warmstarts.items():
        target_path = target_by_name.get(image_name)
        if target_path is None:
            print(f"Skipping {image_name}: target image not found")
            continue

        print(f"\n── {image_name} ──")
        cap = caption_data.get(image_name, {})
        # Warm start stores {warm, ci, subject}; tolerate the legacy plain string.
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
