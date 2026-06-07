"""
Phase 4 – Closed-loop prompt refinement.

Replaces the open-loop Phase 2. For each target image, runs a beam search that
uses the metric feedback:

    warm start (phase1_warmstart.json) -> render + score -> seed the beam
    repeat N_ITERATIONS times:
        show Qwen2.5-VL the TARGET image + the current best RENDER + its scores
        Qwen proposes N_PROPOSALS improved prompts
        render + score each, keep the BEAM_WIDTH best by the branch metric
    final top-3 of the pool = that branch's result

Four branches run per image, differing in (a) a one-line objective hint given to
Qwen and (b) the fitness used to pick survivors:

    clip       – maximise CLIP image-image similarity
    lpips      – minimise LPIPS perceptual distance
    mse        – minimise pixel MSE
    composite  – maximise the standardised average (z(clip) - z(lpips) - z(mse))

The composite uses z-scores so the three scales combine into one continuous
objective. After the loop, every candidate rendered for an image (all branches
pooled) is ranked with Phase 3's Borda composite to produce the submitted top-3,
keeping the deliverable format identical to Phase 3.

Subject vs. style: the warm start prepends the BLIP-2 subject; Qwen is told the
BLIP-2 text is the SUBJECT and the clip-interrogator text is the STYLE / medium.

Negative prompt: a base NSFW/deformity bank always; human-suppression terms for
the non-human targets only. Guidance 8.0 keeps CFG on, so the negative is active.

Input:  phase1_warmstart.json (or phase1_captions.json fallback),
        phase1_captions.json, target images in TARGET_DIR
Output: phase4_results.json   – every rendered candidate, metrics + Borda ranks
        phase4_top3.json/.csv – submitted top-3 per image (Borda composite)
        phase4_branches.json  – best prompt per (image, branch)
        phase4_summary.json   – aggregate stats + metadata for the report
        Renders under outputs/phase4/<image_stem>/<branch>/iterNN_kk.png
        Winning renders copied to outputs/phase4_top3/
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm

from evaluation import evaluate_candidate
from phase2_sampling import load_qwen, parse_numbered_list
from phase3_render_score import (
    TARGET_DIR,
    attach_ranks,
    build_summary,
    default_device,
    load_negative_prompt,
    load_pipeline,
    print_aggregate,
    render,
    seed_from_filename,
)


# ── Config ────────────────────────────────────────────────────────────────────

WARMSTART_FILE  = Path("phase1_warmstart.json")    # from phase1_interrogate.py
PHASE1_CAPTIONS = Path("phase1_captions.json")     # BLIP-2 subject + caption

# P4_TAG suffixes all outputs so a smoke/test run doesn't clobber the real one.
TAG = os.environ.get("P4_TAG", "")

OUTPUT_DIR    = Path(f"outputs/phase4{TAG}")
TOP3_DIR      = Path(f"outputs/phase4{TAG}_top3")
RESULTS_FILE  = Path(f"phase4{TAG}_results.json")
TOP3_JSON     = Path(f"phase4{TAG}_top3.json")
TOP3_CSV      = Path(f"phase4{TAG}_top3.csv")
BRANCHES_FILE = Path(f"phase4{TAG}_branches.json")
SUMMARY_FILE  = Path(f"phase4{TAG}_summary.json")
CHECKPOINT_FILE = Path(f"phase4{TAG}_checkpoint.json")   # resume after a crash/shutdown
TRACE_FILE      = Path(f"phase4{TAG}_trace.jsonl")        # append-only per-call forensics

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"

# Key knobs are env-overridable (so subprocess workers and smoke runs share them).
N_ITERATIONS = int(os.environ.get("P4_ITERS", "12"))      # refinement steps per branch
N_PROPOSALS  = int(os.environ.get("P4_PROPOSALS", "5"))   # prompts proposed per iter
BEAM_WIDTH   = 3      # survivors carried to the next iteration
TOP_K        = 3      # prompts submitted per image
# 5 short prompts fit in ~250 tokens; capping avoids the model rambling to 1024
# (which dominated generation time). Env-overridable.
MAX_NEW_TOKENS = int(os.environ.get("P4_MAXTOK", "320"))
VLM_IMAGE_PX = 512    # downscale images shown to Qwen (VRAM headroom)

# The propose step loops the VLM until N_PROPOSALS (+buffer) distinct prompts are
# gathered, OR PROPOSE_PATIENCE consecutive calls add nothing new, OR PROPOSE_MAX_CALLS
# as a backstop. These are set high on purpose: the model should keep trying hard for
# new wordings (with escalating temperature + a "diverge harder" instruction) before
# we ever declare a branch exhausted. Compute is cheap relative to a flat run.
PROPOSE_PATIENCE  = int(os.environ.get("P4_PATIENCE", "5"))
PROPOSE_MAX_CALLS = int(os.environ.get("P4_MAXCALLS", "12"))
# Renders are deduped by perceptual (dHash) distance: a candidate whose render is
# within IMG_DEDUP_MAX_HAM bits of one already kept is discarded. Default 0 = drop
# ONLY effectively-identical images (same picture up to the coarse hash) — e.g. a
# trailing token the LCM ignores. Genuine variants (different lighting/composition)
# differ by >=1 bit and are kept. Raise it to prune near-duplicates more eagerly.
IMG_DEDUP_MAX_HAM = int(os.environ.get("P4_IMG_HAM", "0"))
# To still land N_PROPOSALS *distinct images* per branch when some renders are
# visual duplicates, the propose step over-generates by IMG_RETRY_BUFFER prompts in
# the same call; the render step renders them only as needed, drawing on the spares
# to replace duplicates. Propose and render are separate processes, so this buffer
# is the refill mechanism (the renderer cannot call back into Qwen).
IMG_RETRY_BUFFER = int(os.environ.get("P4_IMG_BUFFER", "3"))

# Sampling for the proposal step. Diversity comes from the distinct per-(image,
# branch, call) seed, the rotating axis nudge, and — crucially when a branch starts
# repeating itself — a temperature that escalates meaningfully with each stale call
# so the distribution actually flattens and new wordings appear. The ceiling is high
# enough to break out of a peaked distribution; malformed or off-target prompts that
# leak through at high temperature simply score worse and are dropped by selection.
DO_SAMPLE   = True
TEMPERATURE = 0.8
TOP_P       = 0.95
# Optimiser (proposal-sampling) seed. The LCM *render* seed is fixed per image
# (from the filename) and never changes; this only seeds the stochastic prompt
# search. P4_SEED lets the Reporting-Protocol multi-seed sweep (>=5 repetitions)
# vary it without code edits — pair it with a distinct P4_TAG so each repetition
# writes its own outputs/checkpoint instead of clobbering the previous one.
GEN_SEED    = int(os.environ.get("P4_SEED", "1234"))
DIVERSITY_TEMP_STEP = float(os.environ.get("P4_TEMP_STEP", "0.15"))   # +per stale call
DIVERSITY_TEMP_MAX  = float(os.environ.get("P4_TEMP_MAX", "1.20"))

# Each proposal call is nudged to vary one description axis, rotating per call, so
# the batch spreads out instead of clustering on the same phrasing.
DIVERSITY_AXES = [
    "the subject's form, pose and salient details",
    "the colour palette and saturation",
    "the lighting, contrast and time of day",
    "the composition, framing and camera angle",
    "the artistic medium and rendering style",
    "the background and surrounding context",
]

# Both the LCM text encoder and the CLIP scorer cap at 77 tokens (~75 usable);
# anything past that is silently ignored. Prompts are capped shorter so every
# token is in the visible budget and tail-only edits collapse to one prompt.
CLIP_MODEL_NAME    = "openai/clip-vit-large-patch14"
CLIP_PROMPT_TOKENS = int(os.environ.get("P4_CLIP_TOKENS", "60"))
_CLIP_TOKENIZER = None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Branches: name -> (objective hint shown to Qwen, fitness "key" used to select).
# Fitness is computed by select_best() / branch_score(); the hint only steers
# what Qwen emphasises when describing the target.
BRANCHES = {
    "clip":      "Prioritise overall semantic and stylistic similarity to the target.",
    "lpips":     "Prioritise perceptual similarity — textures, structures and mid-level detail.",
    "mse":       "Prioritise exact pixel match — get colours, brightness and the spatial layout/position precisely right.",
    "composite": "Balance semantic, perceptual and exact-pixel similarity to the target.",
}
# P4_BRANCHES (comma list) optionally restricts which branches run.
if os.environ.get("P4_BRANCHES"):
    _keep = {b.strip() for b in os.environ["P4_BRANCHES"].split(",")}
    BRANCHES = {k: v for k, v in BRANCHES.items() if k in _keep}

# Negative prompt. The fixed generation setup the TARGETS were produced under
# (model, seed, 8 steps, guidance 8.0, lcm_origin_steps 50, 768x768) lists NO
# negative prompt, so adding clip-interrogator's negative.txt is a deviation that
# systematically pushes our renders off the targets' manifold and can only hurt
# the image-side metrics. Default OFF to match the targets; P4_NEGATIVE=1 re-enables
# it for an A/B. (It is not the cause of low prompt diversity — that is a
# text-generation issue fixed by the per-call seed/temperature/axis — but removing
# it is the right call for fidelity.)
USE_NEGATIVE = os.environ.get("P4_NEGATIVE", "0") == "1"


# ── Qwen refinement prompt ──────────────────────────────────────────────────────

REFINE_SYSTEM = """\
You are a prompt engineer for the DreamShaper v7 Stable Diffusion model \
(LCM sampling, 8 steps, guidance scale 8.0, fixed seed). You are shown a TARGET \
image to reproduce and the GENERATED image made by the current prompt. Write \
improved prompts that push the generated image closer to the target.

Rules:
- Describe ONLY what is visible in the TARGET: subject, colours, composition, \
background, style, lighting. Never invent content or artists not implied by the target.
- Put the SUBJECT first — it is the most important token.
- Use short comma-separated tags, not full sentences. Keep each prompt within ~60 \
CLIP tokens (about 18 short tags); the renderer ignores anything past that, so do \
not pad with trailing detail.
- Make the variations genuinely different from each other and from prompts already \
tried: change the SUBJECT wording, palette, lighting, composition or medium — never \
just append or swap a trailing adjective.

Output EXACTLY {n} prompts, numbered "1." through "{n}.". No preamble, no commentary."""


def build_user_text(
    subject: str,
    style_hint: str,
    objective: str,
    current_prompt: str,
    metrics: dict,
    tried: list[str],
    n: int,
    axis: str = "",
    push: int = 0,
) -> str:
    """User-message text: states subject (BLIP-2) vs style (CI) roles explicitly,
    the current scores, the branch objective, and what's already been tried. `push`
    rises each time the previous call added nothing new -> a stronger instruction to
    rewrite from scratch rather than nudge."""
    tried_block = "\n".join(f"  - {p}" for p in tried[-20:]) or "  (none yet)"
    style_line = f"\nSTYLE / medium hints (clip-interrogator): {style_hint}" if style_hint else ""
    axis_line = f"\nFor THIS set, especially vary how you describe: {axis}." if axis else ""
    push_line = (
        "\n\nIMPORTANT: your last attempts mostly repeated prompts already tried. "
        "Do NOT lightly edit the current prompt. Re-describe the target FROM SCRATCH "
        "with different sentence structure, synonyms, tag ordering, and emphasis — "
        "the wordings must be clearly distinct from every prompt listed above."
    ) if push else ""
    return (
        f"SUBJECT of the target (from BLIP-2, treat as the main object/scene): "
        f"{subject or 'unknown'}"
        f"{style_line}\n\n"
        f"The first image is the TARGET. The second image is the CURRENT GENERATED "
        f"result from this prompt:\n  \"{current_prompt}\"\n\n"
        f"Current similarity scores (higher CLIP = better; lower LPIPS / MSE = better):\n"
        f"  CLIP={metrics['clip_similarity']:.4f}  "
        f"LPIPS={metrics['lpips']:.4f}  MSE={metrics['pixel_mse']:.5f}\n\n"
        f"Objective for these variations: {objective}{axis_line}\n\n"
        f"Prompts already tried (do NOT repeat these):\n{tried_block}{push_line}\n\n"
        f"Treat the BLIP-2 SUBJECT as what the thing IS, and the clip-interrogator "
        f"text as how it LOOKS (style/medium). Generate {n} improved, distinct prompts."
    )


def clip_truncate(prompt: str, max_tokens: int = CLIP_PROMPT_TOKENS) -> str:
    """Truncate a prompt to at most max_tokens CLIP tokens. The LCM text encoder
    and the CLIP scorer both cap at 77 tokens; capping shorter keeps every token
    visible and makes dedup honest (edits past the cap collapse to one prompt)."""
    global _CLIP_TOKENIZER
    p = " ".join((prompt or "").split())
    if not p:
        return p
    if _CLIP_TOKENIZER is None:
        from transformers import CLIPTokenizerFast
        _CLIP_TOKENIZER = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)
    ids = _CLIP_TOKENIZER(p, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return p
    return _CLIP_TOKENIZER.decode(ids[:max_tokens]).strip().rstrip(",.;: ").strip()


def _vlm_image(path: Path):
    """Downscale an image for the VLM (fewer vision tokens -> less VRAM)."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((VLM_IMAGE_PX, VLM_IMAGE_PX))
    return im


def _norm(prompt: str) -> str:
    """Normalise a prompt for exact-duplicate comparison (case/whitespace-insensitive)."""
    return " ".join(prompt.lower().split())


def _phash(path, hash_size: int = 8):
    """Difference hash of a render: a boolean array of length hash_size**2. Two
    visually identical images hash within a few bits (Hamming) of each other."""
    from PIL import Image
    import numpy as np
    resample = getattr(Image, "Resampling", Image).LANCZOS
    im = Image.open(path).convert("L").resize((hash_size + 1, hash_size), resample)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _hamming(a, b) -> int:
    return int((a != b).sum())


def qwen_propose(
    model, processor, target_path: Path, gen_path: Path, user_text: str, n: int,
    temperature: float = TEMPERATURE, seed: int | None = None,
) -> tuple[list[str], str]:
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": REFINE_SYSTEM.format(n=n)},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _vlm_image(target_path)},
                {"type": "image", "image": _vlm_image(gen_path)},
                {"type": "text", "text": user_text},
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
    gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE}
    if DO_SAMPLE:
        gen_kwargs.update(temperature=temperature, top_p=TOP_P)
        if seed is not None:
            torch.manual_seed(seed)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    parsed = parse_numbered_list(output, n)
    if not parsed:
        print(f"   [warn] {target_path.name}: no prompts parsed from VLM output: "
              f"{output[:200]!r}", flush=True)
    return parsed, output


def collect_proposals(model, processor, name, s, b, t, proposals) -> list[str]:
    """Loop the VLM (resident, no model swap) until N_PROPOSALS distinct NEW prompts
    are gathered, or PROPOSE_PATIENCE consecutive calls add nothing new, or the
    PROPOSE_MAX_CALLS backstop. The growing list is fed back as the avoid-list each
    call, and prompts proposed in earlier iterations are excluded too, so we never
    re-offer something already rendered (or already rejected as a visual duplicate)."""
    bp = branch_pool(s["pool"], b)
    current = select_best(b, bp, 1)[0]
    prior = [p for k, v in proposals.items()
             if k.startswith(f"{name}|{b}|") for p in v]   # earlier iterations
    base_tried = [r["prompt"] for r in bp] + prior
    avoid = {_norm(p) for p in base_tried}

    # Gather N_PROPOSALS + a buffer of distinct prompts per call, so the render step
    # has spares to swap in when a render turns out to be a visual duplicate.
    target = N_PROPOSALS + IMG_RETRY_BUFFER
    collected: list[str] = []
    collected_norm: set[str] = set()
    stale = 0
    for call in range(PROPOSE_MAX_CALLS):
        if len(collected) >= target:
            break
        # Escalate temperature while we keep getting nothing new; rotate the axis
        # nudge per call. Seed is reproducible but distinct per (image, branch, call).
        temperature = min(TEMPERATURE + DIVERSITY_TEMP_STEP * stale, DIVERSITY_TEMP_MAX)
        axis = DIVERSITY_AXES[call % len(DIVERSITY_AXES)]
        seed = GEN_SEED + int.from_bytes(
            hashlib.md5(f"{name}|{b}|{call}".encode()).digest()[:4], "big") % 1_000_000
        ut = build_user_text(s["subject"], s["style_hint"], BRANCHES[b],
                             current["prompt"], current, base_tried + collected,
                             target, axis=axis, push=stale)
        batch, raw = qwen_propose(model, processor, s["target_path"],
                                  Path(current["render"]), ut, target,
                                  temperature=temperature, seed=seed)
        new = 0
        truncated = [clip_truncate(p) for p in batch]
        for p in truncated:
            k = _norm(p)
            if not p or k in avoid or k in collected_norm:
                continue
            collected.append(p)
            collected_norm.add(k)
            new += 1
            if len(collected) >= target:
                break
        # One trace line per VLM call: enough to reconstruct exactly which axis /
        # temperature / seed produced which prompts, how many were new, and the raw
        # model text (so parse failures and near-duplicate runs are diagnosable).
        trace_append({
            "event": "propose_call", "image": name, "branch": b, "iter": t,
            "call": call, "axis": axis, "temperature": round(temperature, 3),
            "seed": seed, "n_parsed": len(batch), "n_new": new,
            "proposed": truncated, "raw": raw,
        })
        stale = 0 if new else stale + 1
        if stale >= PROPOSE_PATIENCE:
            break
    return collected


# ── Fitness ───────────────────────────────────────────────────────────────────

def _z(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    m = statistics.mean(values)
    s = statistics.pstdev(values)
    return [0.0] * len(values) if s == 0 else [(v - m) / s for v in values]


def branch_score(branch: str, pool: list[dict]) -> list[float]:
    """Per-candidate fitness for `branch`, HIGHER = better, aligned to pool order."""
    if branch == "clip":
        return [r["clip_similarity"] for r in pool]
    if branch == "lpips":
        return [-r["lpips"] for r in pool]
    if branch == "mse":
        return [-r["pixel_mse"] for r in pool]
    # composite: standardised average, lower lpips/mse better -> negate their z
    zc = _z([r["clip_similarity"] for r in pool])
    zl = _z([r["lpips"] for r in pool])
    zm = _z([r["pixel_mse"] for r in pool])
    return [zc[i] - zl[i] - zm[i] for i in range(len(pool))]


def select_best(branch: str, pool: list[dict], k: int) -> list[dict]:
    scores = branch_score(branch, pool)
    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    return [pool[i] for i in order[:k]]


# ── Per-image refinement ────────────────────────────────────────────────────────

def score_prompt(
    prompt, seed, neg, pipe, device, target_path, render_path, branch, iteration, idx
) -> dict:
    if not render_path.exists():
        render(prompt, seed, pipe, device, negative_prompt=neg).save(render_path)
    metrics = evaluate_candidate(target_path, render_path)
    return {
        "target": str(target_path),
        "target_name": target_path.name,
        "render_seed": seed,
        "candidate_index": idx,
        "branch": branch,
        "iteration": iteration,
        "prompt": prompt,
        "render": str(render_path),
        **metrics,
    }


def free_gpu():
    """Release GPU memory. Caller must `del` its own references FIRST — deleting
    a parameter here wouldn't drop the caller's binding."""
    gc.collect()
    torch.cuda.empty_cache()


def trace_append(record: dict):
    """Append one JSON line to the run trace. Append-only and flushed per write so
    a crash leaves a complete record up to the last event. Used for per-call
    proposal forensics (axis/temp/seed/raw output) and dedup events — none of which
    the checkpoint captures — so the diversity mechanism can be analysed offline."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with TRACE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(state, proposals, converged=None):
    data = {
        "pool_by_image": {n: s["pool"] for n, s in state.items()},
        "proposals": proposals,
        "converged": sorted(converged or []),
    }
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(CHECKPOINT_FILE)   # atomic: a power-off won't corrupt the file


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        d = json.loads(CHECKPOINT_FILE.read_text())
        return (d.get("pool_by_image", {}), d.get("proposals", {}),
                set(d.get("converged", [])))
    return {}, {}, set()


# ── Main ──────────────────────────────────────────────────────────────────────

def load_warmstarts() -> dict:
    """Return name -> {"warm", "ci", "subject"}. Accepts the phase1_interrogate
    dict format, or falls back to phase1_captions assembled_prompt."""
    if WARMSTART_FILE.exists():
        data = json.loads(WARMSTART_FILE.read_text())
        out = {}
        for name, e in data.items():
            if isinstance(e, dict):
                out[name] = {"warm": e.get("warm", ""), "ci": e.get("ci", ""),
                             "subject": e.get("subject", "")}
            else:  # legacy: plain warm-start string
                out[name] = {"warm": e, "ci": "", "subject": ""}
        return out
    print(f"  ! {WARMSTART_FILE} not found — falling back to phase1_captions "
          f"assembled_prompt.")
    caps = json.loads(PHASE1_CAPTIONS.read_text())
    return {
        name: {
            "warm": e.get("assembled_prompt") or e.get("subject", ""),
            "ci": "",
            "subject": e.get("subject", ""),
        }
        for name, e in caps.items()
    }


def branch_pool(pool: list[dict], branch: str) -> list[dict]:
    """Candidates a branch selects over: its own rows plus the shared warm start."""
    return [r for r in pool if r["branch"] in ("warm", branch)]


def build_state() -> dict:
    """Static per-image info (no pool). Workers rebuild this deterministically."""
    warmstarts = load_warmstarts()
    captions = (json.loads(PHASE1_CAPTIONS.read_text())
                if PHASE1_CAPTIONS.exists() else {})
    target_by_name = {
        p.name: p for p in TARGET_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    limit = int(os.environ.get("P4_LIMIT", "0"))   # 0 = all images
    state = {}
    for i, (name, ws) in enumerate(warmstarts.items()):
        if limit and i >= limit:
            break
        tp = target_by_name.get(name)
        if tp is None:
            continue
        cap = captions.get(name, {})
        base = OUTPUT_DIR / Path(name).stem
        for b in list(BRANCHES) + ["_warm"]:
            (base / b).mkdir(parents=True, exist_ok=True)
        state[name] = {
            "target_path": tp,
            "seed": seed_from_filename(tp),
            "subject": ws.get("subject") or cap.get("subject", ""),
            "style_hint": ws.get("ci") or cap.get("base_caption", ""),
            "warm_prompt": ws["warm"],
            "base": base,
            "pool": [],
        }
    return state


def _load_state_with_pools():
    state = build_state()
    saved_pools, proposals, converged = load_checkpoint()
    for name in state:
        state[name]["pool"] = saved_pools.get(name, [])
    return state, proposals, converged


def _all_converged() -> bool:
    """True once every in-scope image has stopped producing new proposals."""
    state, _, converged = _load_state_with_pools()
    return bool(state) and all(n in converged for n in state)


# ── Worker: render phase (own process → clean GPU, pipe gets full VRAM) ─────────

def run_render(t: int):
    state, proposals, converged = _load_state_with_pools()
    device = default_device()
    negative = load_negative_prompt() if USE_NEGATIVE else ""
    done = {r["render"] for s in state.values() for r in s["pool"]}
    pipe = load_pipeline(device)
    for name, s in state.items():
        if t >= 1 and name in converged:
            continue
        if t == 0:   # warm-start render
            if any(r["branch"] == "warm" for r in s["pool"]):
                continue
            rp = s["base"] / "_warm" / "warm.png"
            row = score_prompt(clip_truncate(s["warm_prompt"]), s["seed"], negative,
                               pipe, device,
                               s["target_path"], rp, "warm", 0, len(s["pool"]) + 1)
            s["pool"].append(row); done.add(row["render"])
            print(f"   warm {name}: CLIP={row['clip_similarity']:.4f}", flush=True)
            continue
        for b in BRANCHES:
            # Per-branch namespace: dedup only against this branch (+ the shared warm
            # start), so branches never suppress each other's candidates.
            bp = branch_pool(s["pool"], b)
            seen = {_norm(r["prompt"]) for r in bp}
            kept_hashes = [_phash(p) for p in
                           (Path(r["render"]) for r in bp) if p.exists()]
            # Stop once this branch has N_PROPOSALS distinct *images* this iteration;
            # buffered prompts beyond that are only rendered to replace duplicates.
            # Rows already kept (e.g. on resume) count toward the target.
            kept_this_iter = sum(1 for r in s["pool"]
                                 if r["branch"] == b and r["iteration"] == t)
            for j, prompt in enumerate(proposals.get(f"{name}|{b}|{t}", []), start=1):
                if kept_this_iter >= N_PROPOSALS:
                    break
                rp = s["base"] / b / f"iter{t:02d}_{j:02d}.png"
                kp = _norm(prompt)
                if str(rp) in done or not prompt or kp in seen:
                    continue
                if not rp.exists():
                    render(prompt, s["seed"], pipe, device,
                           negative_prompt=negative).save(rp)
                h = _phash(rp)
                ham = min((_hamming(h, k) for k in kept_hashes), default=None)
                if ham is not None and ham <= IMG_DEDUP_MAX_HAM:
                    # Visual duplicate -> not a new candidate. Move (not delete) the
                    # render to a _dupes/ sibling and log it, so what the dedup filtered
                    # stays auditable for the diversity analysis.
                    dup_dir = s["base"] / "_dupes" / b
                    dup_dir.mkdir(parents=True, exist_ok=True)
                    dup_path = dup_dir / rp.name
                    rp.replace(dup_path)
                    seen.add(kp)
                    trace_append({
                        "event": "render_dedup", "image": name, "branch": b, "iter": t,
                        "prompt": prompt, "dup_render": str(dup_path), "min_hamming": ham,
                    })
                    continue
                row = score_prompt(prompt, s["seed"], negative, pipe, device,
                                   s["target_path"], rp, b, t, len(s["pool"]) + 1)
                s["pool"].append(row); done.add(row["render"])
                seen.add(kp); kept_hashes.append(h); kept_this_iter += 1
        tops = " ".join(
            f"{b}={select_best(b, branch_pool(s['pool'], b), 1)[0]['clip_similarity']:.3f}"
            for b in BRANCHES)
        print(f"   iter {t} {name}: CLIP best -> {tops}", flush=True)
    save_checkpoint(state, proposals, converged)


# ── Worker: propose phase (own process → Qwen gets full VRAM, uncapped) ─────────

def run_propose(t: int):
    state, proposals, converged = _load_state_with_pools()
    model, processor = load_qwen(QWEN_MODEL_ID)
    for name, s in state.items():
        if name in converged:
            continue
        new_total = 0
        for b in BRANCHES:
            key = f"{name}|{b}|{t}"
            if key not in proposals:
                proposals[key] = collect_proposals(model, processor, name, s, b, t, proposals)
                print(f"   iter {t} {name} [{b}]: {len(proposals[key])} distinct proposals",
                      flush=True)
                save_checkpoint(state, proposals, converged)
            new_total += len(proposals[key])
        # No branch found anything new this iteration -> the subject is exhausted;
        # stop refining this image so we don't reload models for empty work.
        if new_total == 0:
            converged.add(name)
            print(f"   iter {t} {name}: converged (no new proposals), "
                  f"skipping further iterations", flush=True)
            save_checkpoint(state, proposals, converged)


# ── Orchestrator: run each phase as a subprocess (clean GPU every time) ─────────

def _run_phase(worker: str, t: int):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", worker, "--iter", str(t)]
    print(f"\n>>> {worker} iter {t} (subprocess)", flush=True)
    res = subprocess.run(cmd, env=os.environ.copy())
    if res.returncode != 0:
        raise SystemExit(f"{worker} iter {t} failed (exit {res.returncode}); "
                         f"checkpoint preserved — re-run to resume.")


def _pick_distinct_top(rows: list[dict], k: int) -> list[dict]:
    """Pick the k best rows (rows are already Borda-sorted) whose renders are
    visually distinct from one another. Per-branch dedup lets the same image win
    under several branches, so the pooled top can hide duplicates; this filters
    them at submission. If fewer than k visually-distinct renders exist, pad with
    the next-best (deduped) rows so the deliverable always has k entries."""
    chosen, chosen_h, leftover = [], [], []
    for r in rows:
        p = Path(r["render"])
        h = _phash(p) if p.exists() else None
        if h is not None and chosen_h and \
                min(_hamming(h, c) for c in chosen_h) <= IMG_DEDUP_MAX_HAM:
            leftover.append(r)
            continue
        chosen.append(r)
        if h is not None:
            chosen_h.append(h)
        if len(chosen) >= k:
            return chosen
    for r in leftover:                      # genuinely few distinct -> pad to k
        if len(chosen) >= k:
            break
        chosen.append(r)
    return chosen[:k]


def finalize():
    state, _, _ = _load_state_with_pools()
    device = default_device()
    all_results: dict[str, list[dict]] = {}
    all_branches: dict[str, dict] = {}
    for name, s in state.items():
        pool = list(s["pool"])
        attach_ranks(pool)
        all_results[name] = pool
        all_branches[name] = {}
        for b in BRANCHES:
            best = select_best(b, branch_pool(pool, b), 1)[0]
            all_branches[name][b] = {
                "prompt": best["prompt"], "render": best["render"],
                "clip_similarity": round(best["clip_similarity"], 6),
                "lpips": round(best["lpips"], 6),
                "pixel_mse": round(best["pixel_mse"], 6),
                "pixel_rmse": round(best["pixel_rmse"], 6),
            }

    RESULTS_FILE.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    BRANCHES_FILE.write_text(json.dumps(all_branches, indent=2, ensure_ascii=False))
    print(f"\nSaved all candidates + ranks to {RESULTS_FILE}")
    print(f"Saved per-branch winners to {BRANCHES_FILE}")

    # ── Deliverable: top-3 per image (Borda composite, same format as phase 3) ──
    TOP3_DIR.mkdir(parents=True, exist_ok=True)
    top3_rows: list[dict] = []
    for image_name, rows in all_results.items():
        stem = Path(image_name).stem
        for rank, r in enumerate(_pick_distinct_top(rows, TOP_K), start=1):
            dest = TOP3_DIR / f"{stem}_rank{rank}_{r['branch']}_cand{r['candidate_index']:03d}.png"
            shutil.copyfile(r["render"], dest)
            top3_rows.append({**r, "submission_rank": rank, "top3_render": str(dest)})

    TOP3_JSON.write_text(json.dumps(top3_rows, indent=2, ensure_ascii=False))
    fields = [
        "target", "target_name", "render_seed", "submission_rank", "branch",
        "iteration", "candidate_index", "prompt", "render", "top3_render",
        "clip_similarity", "lpips", "pixel_mse", "pixel_rmse",
        "rank_clip", "rank_lpips", "rank_mse", "rank_composite",
    ]
    with TOP3_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(top3_rows)
    print(f"Saved top-{TOP_K} deliverable to {TOP3_JSON} and {TOP3_CSV}")
    print(f"Copied {len(top3_rows)} winning renders to {TOP3_DIR}/")

    # ── Aggregate stats ────────────────────────────────────────────────────────
    best1 = [rows[0] for rows in all_results.values()]
    print("\n" + "=" * 70)
    print("AGGREGATE METRICS ACROSS THE TEST SET")
    print("=" * 70)
    print_aggregate("Best prompt per image (top-1)", best1)
    print_aggregate(f"Submitted prompts (top-{TOP_K})", top3_rows)

    dtype = "float16" if device == "cuda" else "float32"
    summary = build_summary(all_results, top3_rows, device, dtype)
    summary["metadata"]["phase"] = "phase4_refine"
    summary["metadata"]["n_iterations"] = N_ITERATIONS
    summary["metadata"]["n_proposals"] = N_PROPOSALS
    summary["metadata"]["beam_width"] = BEAM_WIDTH
    summary["metadata"]["branches"] = list(BRANCHES.keys())
    summary["metadata"]["vlm"] = QWEN_MODEL_ID
    summary["metadata"]["do_sample"] = DO_SAMPLE
    summary["metadata"]["temperature"] = TEMPERATURE if DO_SAMPLE else None
    summary["metadata"]["top_p"] = TOP_P if DO_SAMPLE else None
    summary["metadata"]["gen_seed"] = GEN_SEED
    summary["metadata"]["negative_prompt"] = "clip-interrogator" if USE_NEGATIVE else "none"
    summary["metadata"]["diversity"] = {
        "propose_until_n_distinct": N_PROPOSALS,
        "propose_patience": PROPOSE_PATIENCE,
        "propose_max_calls": PROPOSE_MAX_CALLS,
        "temperature_base": TEMPERATURE,
        "temperature_step": DIVERSITY_TEMP_STEP,
        "temperature_max": DIVERSITY_TEMP_MAX,
        "image_dedup_max_hamming": IMG_DEDUP_MAX_HAM,
        "method": "convergence-guarded proposal loop + per-branch image-level dedup",
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved report summary to {SUMMARY_FILE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["propose", "render"])
    ap.add_argument("--iter", type=int, default=0)
    args = ap.parse_args()

    # Worker mode: do one phase in this (fresh) process, then exit → GPU freed.
    if args.worker == "propose":
        run_propose(args.iter)
        return
    if args.worker == "render":
        run_render(args.iter)
        return

    # Orchestrator: run each phase as its own subprocess so the VLM and the
    # diffusion pipeline never share VRAM and each starts with a clean CUDA
    # context. Everything flows through the checkpoint, so a crash/power-off
    # just means "re-run to resume".
    print(f"Phase 4 closed-loop: {len(BRANCHES)} branches x {N_ITERATIONS} iters")
    _run_phase("render", 0)                       # warm-start renders
    for t in range(1, N_ITERATIONS + 1):
        _run_phase("propose", t)
        _run_phase("render", t)
        if _all_converged():
            print(f"\nAll images converged by iter {t}; stopping early.", flush=True)
            break
    finalize()
    print("\nDONE.")


if __name__ == "__main__":
    main()
