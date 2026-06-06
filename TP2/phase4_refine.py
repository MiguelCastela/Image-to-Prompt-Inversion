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

# Sampling for the proposal step: temperature > 0 is required for the N_PROPOSALS
# variants (and successive iterations) to differ. GEN_SEED fixes the RNG once at
# the start of the run so the whole sequence is reproducible.
DO_SAMPLE   = True
TEMPERATURE = 0.7
TOP_P       = 0.9
GEN_SEED    = 1234

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

# Negative prompt is loaded via phase3's load_negative_prompt() (clip-interrogator
# negative.txt, comma-joined). CFG is on at guidance 8.0, so it is active.


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
- Use short comma-separated tags, not full sentences. Keep each prompt under 30 words.
- Make the variations genuinely different from each other and from prompts already tried.

Output EXACTLY {n} prompts, numbered "1." through "{n}.". No preamble, no commentary."""


def build_user_text(
    subject: str,
    style_hint: str,
    objective: str,
    current_prompt: str,
    metrics: dict,
    tried: list[str],
    n: int,
) -> str:
    """User-message text: states subject (BLIP-2) vs style (CI) roles explicitly,
    the current scores, the branch objective, and what's already been tried."""
    tried_block = "\n".join(f"  - {p}" for p in tried[-15:]) or "  (none yet)"
    style_line = f"\nSTYLE / medium hints (clip-interrogator): {style_hint}" if style_hint else ""
    return (
        f"SUBJECT of the target (from BLIP-2, treat as the main object/scene): "
        f"{subject or 'unknown'}"
        f"{style_line}\n\n"
        f"The first image is the TARGET. The second image is the CURRENT GENERATED "
        f"result from this prompt:\n  \"{current_prompt}\"\n\n"
        f"Current similarity scores (higher CLIP = better; lower LPIPS / MSE = better):\n"
        f"  CLIP={metrics['clip_similarity']:.4f}  "
        f"LPIPS={metrics['lpips']:.4f}  MSE={metrics['pixel_mse']:.5f}\n\n"
        f"Objective for these variations: {objective}\n\n"
        f"Prompts already tried (do NOT repeat these):\n{tried_block}\n\n"
        f"Treat the BLIP-2 SUBJECT as what the thing IS, and the clip-interrogator "
        f"text as how it LOOKS (style/medium). Generate {n} improved, distinct prompts."
    )


def _vlm_image(path: Path):
    """Downscale an image for the VLM (fewer vision tokens -> less VRAM)."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((VLM_IMAGE_PX, VLM_IMAGE_PX))
    return im


def qwen_propose(
    model, processor, target_path: Path, gen_path: Path, user_text: str, n: int
) -> list[str]:
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
        gen_kwargs.update(temperature=TEMPERATURE, top_p=TOP_P)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return parse_numbered_list(output, n)


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


def save_checkpoint(state, proposals):
    data = {
        "pool_by_image": {n: s["pool"] for n, s in state.items()},
        "proposals": proposals,
    }
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(CHECKPOINT_FILE)   # atomic: a power-off won't corrupt the file


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        d = json.loads(CHECKPOINT_FILE.read_text())
        return d.get("pool_by_image", {}), d.get("proposals", {})
    return {}, {}


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
    saved_pools, proposals = load_checkpoint()
    for name in state:
        state[name]["pool"] = saved_pools.get(name, [])
    return state, proposals


# ── Worker: render phase (own process → clean GPU, pipe gets full VRAM) ─────────

def run_render(t: int):
    state, proposals = _load_state_with_pools()
    device = default_device()
    negative = load_negative_prompt()
    done = {r["render"] for s in state.values() for r in s["pool"]}
    pipe = load_pipeline(device)
    for name, s in state.items():
        if t == 0:   # warm-start render
            if any(r["branch"] == "warm" for r in s["pool"]):
                continue
            rp = s["base"] / "_warm" / "warm.png"
            row = score_prompt(s["warm_prompt"], s["seed"], negative, pipe, device,
                               s["target_path"], rp, "warm", 0, len(s["pool"]) + 1)
            s["pool"].append(row); done.add(row["render"])
            print(f"   warm {name}: CLIP={row['clip_similarity']:.4f}", flush=True)
            continue
        for b in BRANCHES:
            seen = {r["prompt"] for r in s["pool"]}
            for j, prompt in enumerate(proposals.get(f"{name}|{b}|{t}", []), start=1):
                rp = s["base"] / b / f"iter{t:02d}_{j:02d}.png"
                if str(rp) in done or not prompt or prompt in seen:
                    continue
                row = score_prompt(prompt, s["seed"], negative, pipe, device,
                                   s["target_path"], rp, b, t, len(s["pool"]) + 1)
                s["pool"].append(row); done.add(row["render"]); seen.add(prompt)
        tops = " ".join(
            f"{b}={select_best(b, branch_pool(s['pool'], b), 1)[0]['clip_similarity']:.3f}"
            for b in BRANCHES)
        print(f"   iter {t} {name}: CLIP best -> {tops}", flush=True)
    save_checkpoint(state, proposals)


# ── Worker: propose phase (own process → Qwen gets full VRAM, uncapped) ─────────

def run_propose(t: int):
    state, proposals = _load_state_with_pools()
    model, processor = load_qwen(QWEN_MODEL_ID)
    for name, s in state.items():
        for b in BRANCHES:
            key = f"{name}|{b}|{t}"
            if key in proposals:
                continue
            bp = branch_pool(s["pool"], b)
            current = select_best(b, bp, 1)[0]
            tried = [r["prompt"] for r in bp]
            ut = build_user_text(s["subject"], s["style_hint"], BRANCHES[b],
                                 current["prompt"], current, tried, N_PROPOSALS)
            proposals[key] = qwen_propose(
                model, processor, s["target_path"], Path(current["render"]),
                ut, N_PROPOSALS)
            print(f"   iter {t} {name} [{b}]: {len(proposals[key])} proposals", flush=True)
            save_checkpoint(state, proposals)


# ── Orchestrator: run each phase as a subprocess (clean GPU every time) ─────────

def _run_phase(worker: str, t: int):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", worker, "--iter", str(t)]
    print(f"\n>>> {worker} iter {t} (subprocess)", flush=True)
    res = subprocess.run(cmd, env=os.environ.copy())
    if res.returncode != 0:
        raise SystemExit(f"{worker} iter {t} failed (exit {res.returncode}); "
                         f"checkpoint preserved — re-run to resume.")


def finalize():
    state, _ = _load_state_with_pools()
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
        for rank, r in enumerate(rows[:TOP_K], start=1):
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
    finalize()
    print("\nDONE.")


if __name__ == "__main__":
    main()
