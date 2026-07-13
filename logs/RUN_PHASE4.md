# Phase 4 re-run — runbook for the GPU machine

Run this on the box with the GPU (`jmrc-ASUS`), not the low-VRAM laptop. Every
command is copy-paste ready. The whole point is to regenerate the Phase-4
deliverables with the fixes applied, because the current `phase4_*.json/csv` are
from the **pre-fix** night run (converged at iteration 1, temp 0.7, 7836 had only
1 candidate).

## What changed since that night run (already committed in `phase4_refine.py`)

1. **CLIP-token cap** — prompts truncated to ≤60 CLIP tokens before render/score,
   so the scorer actually sees every token (it silently ignores past 77).
2. **Per-image early-stop** — an image that yields no new distinct proposal in an
   iteration is marked converged and skipped; the run stops when all converge.
   This is what makes a full run finish in well under the old ~7 h.
3. **Diversity** — temperature escalates (0.8→1.3) while a branch keeps returning
   nothing new, a description axis rotates per call, and each VLM call gets a
   distinct reproducible seed. Fixes the "no diversity" stall.
4. **7836 parse logging** — raw VLM output is printed when nothing parses, so the
   zero-proposal image is diagnosable instead of silent.
5. **`P4_SEED`** — the optimiser seed is now env-overridable for the multi-seed
   sweep. The LCM **render** seed is still fixed per image (from the filename).
6. **Negative prompt OFF by default** — the targets' fixed setup lists no negative
   prompt, so we no longer add one (it was a deviation that pushed renders off the
   targets). `P4_NEGATIVE=1` re-enables it for an A/B. This changes results vs. the
   old run, which had it ON.
7. **Per-call trace** — each run writes `phase4<tag>_trace.jsonl` (one line per VLM
   call: axis, temperature, seed, proposed prompts, raw model text, and dedup
   events) and **moves** deduped duplicate renders to `outputs/phase4<tag>/<image>/
   _dupes/<branch>/` instead of deleting them. Both exist purely for offline
   analysis of the diversity mechanism.

Shared inputs reused as-is (do NOT regenerate): `phase1_warmstart.json`,
`phase1_captions.json`. The negative prompt loads from the installed
`clip_interrogator` package, so no `negative.txt` is needed in the repo.

## Pre-flight

```bash
cd ~/Documents/GitHub/Gen-AI/TP2     # adjust to wherever the repo is on that box
git pull                             # get the committed fixes + these scripts
source .venv/bin/activate
pkill -f phase4_refine.py || true    # make sure no stale GPU process is running
```

## Option A — one clean run (the must-have headline deliverable)

A fresh tag isolates the new outputs and preserves the old run as a fallback.

```bash
SEEDS="1234" ./run_phase4_sweep.sh
```

Produces (note the `_s1234` tag):
- `phase4_s1234_top3.csv` / `.json` — top-3 prompts per image, ranked best→worst,
  with the per-candidate table (CLIP image-image, LPIPS, pixel MSE).
- `phase4_s1234_summary.json` — mean ± std across the set.
- `outputs/phase4_s1234_top3/` — the rendered winners with readable names.
- `phase4_s1234.log` — full console log (check the tail).

## Option B — full Reporting-Protocol sweep (≥5 seeds)

The prompt search is stochastic, so the statement requires repeating with at
least 5 seeds and reporting mean ± std across them (render seed stays fixed).

```bash
./run_phase4_sweep.sh                  # seeds 1234 5678 9012 3456 7890, 12 iters each
# tight on time? shorten each repetition:
P4_ITERS=6 ./run_phase4_sweep.sh
```

After all repetitions it auto-runs the aggregator and writes
`phase4_seed_aggregate.json` (mean ± std of each metric **across the 5 runs**) —
that is the table the report's "across the full set" section needs. You can also
run it by hand any time:

```bash
python phase4_aggregate_seeds.py       # globs phase4_s*_summary.json
```

## If a run crashes or the box reboots

Re-run the **exact same** command. Each tag has its own
`phase4_s<seed>_checkpoint.json`; the run resumes from where it stopped (atomic
checkpoint, so a power-off won't corrupt it). Converged images stay skipped.

## Speed knobs (env vars, no code edits)

| Var | Default | Effect |
|-----|---------|--------|
| `P4_ITERS` | 12 | refinement iterations per branch |
| `P4_BRANCHES` | all 4 | restrict, e.g. `composite,clip` |
| `P4_SEED` | 1234 | optimiser seed (set per repetition by the sweep) |
| `P4_TAG` | "" | output suffix (set per repetition by the sweep) |
| `P4_TEMP_MAX` | 0.85 | ceiling for diversity temperature escalation |
| `P4_CLIP_TOKENS` | 60 | prompt truncation budget |
| `P4_NEGATIVE` | 0 | `1` re-enables the clip-interrogator negative prompt |

## Morning review checklist

1. Tail each `phase4_s*.log` — confirm images converge (not stall at iter 1) and
   that **7836 now prints >0 distinct proposals** (the bug we fixed).
2. Open `phase4_s1234_top3.csv` — sanity-check the top-3 prompts and metrics.
3. Open `phase4_seed_aggregate.json` — mean ± std across seeds for the report.
4. Diversity audit (optional): `phase4_s1234_trace.jsonl` has every VLM call's
   axis/temp/seed + raw text + dedup events; `outputs/phase4_s1234/<image>/_dupes/`
   holds the renders the dedup filtered. Use these for the report's diversity
   discussion.
5. Paste the log tails back here and we review together.
