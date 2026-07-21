# Image-to-Prompt Inversion

Recover the text prompt behind an image. Given a target image and a fixed
text-to-image generator (`SimianLuo/LCM_Dreamshaper_v7`), this project searches
for a prompt that, when rendered, reproduces the target as closely as possible.

The method has two stages:

1. **Warm start.** BLIP-2 produces a subject and caption, and CLIP-Interrogator
   builds an initial prompt.
2. **Closed-loop refinement.** A Qwen2.5-VL prompt sampler proposes edits, each
   candidate is rendered with the LCM generator, scored against the target with
   CLIP, LPIPS, and pixel MSE/RMSE, and the best prompts are kept and iterated.
   Results are aggregated across several optimiser seeds for stability.

All model checkpoints are open weights and download automatically on first run.
A CUDA-capable GPU and Python 3.10 or newer are required.

## Repository layout

```
.
├── src/                        the inversion pipeline (run from the repo root)
│   ├── vlm_caption.py              Stage 1: BLIP-2 subject and caption
│   ├── phase1_interrogate.py       Stage 1: CLIP-Interrogator warm-start prompt
│   ├── phase1_clip.py              Stage 1: CLIP scoring helpers
│   ├── phase2_sampling.py          shared Qwen loading and prompt parsing
│   ├── phase3_render_score.py      shared LCM rendering, ranking, summaries
│   ├── phase4_refine.py            Stage 2: closed-loop refinement and selection
│   ├── phase4_aggregate_seeds.py   aggregate metrics across optimiser seeds
│   ├── evaluation.py               metrics (CLIP, LPIPS, pixel MSE/RMSE)
│   └── main.py                     pipeline entry point
├── analysis/                   report-building and post-processing scripts
│   ├── compute_table2.py, compute_annexB.py, compute_4seed_analysis.py
│   ├── make_charts.py              figures for the report
│   ├── build_section.py            inject computed results into the LaTeX
│   └── strip_and_deliver.py        build the comment-stripped delivery/src
├── delivery/                   self-contained submission (code, top-3, PDF)
│   ├── README.md                   standalone run instructions for the delivery
│   ├── requirements.txt            pinned dependencies for the pipeline
│   ├── src/                        the pipeline with comments removed
│   └── top3/                       submitted top-3 prompts and renders per seed
├── logs/                       JSON/CSV metrics, traces, and run logs per seed
├── outputs/                    rendered candidate images (bulky, mostly ignored)
├── report/                     LaTeX report, figures, and compiled main.pdf
└── statement/                  assignment brief and target images
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r delivery/requirements.txt
```

Model checkpoints (the LCM generator, BLIP-2, the CLIP encoders, and the
Qwen2.5-VL refiner) download automatically on first run. No API key is needed for
the pipeline.

## Running the pipeline

Run every command from the repository root so the relative input and output
paths resolve correctly. The pipeline modules live in `src/` but are invoked from
the root:

```
python src/vlm_caption.py                                  # Stage 1: caption
python src/phase1_interrogate.py                           # Stage 1: warm-start prompt
P4_SEED=1234 P4_TAG=_s1234 python src/phase4_refine.py     # Stage 2: refine (repeat per seed)
python src/phase4_aggregate_seeds.py                       # aggregate across seeds
```

For the multi-seed protocol, repeat the refinement step with each seed, changing
both variables together:

```
P4_SEED=3456 P4_TAG=_s3456 python src/phase4_refine.py
P4_SEED=5678 P4_TAG=_s5678 python src/phase4_refine.py
P4_SEED=9012 P4_TAG=_s9012 python src/phase4_refine.py
P4_SEED=7890 P4_TAG=_s7890 python src/phase4_refine.py
```

The LCM render seed stays fixed per image; only `P4_SEED`, which controls the
Qwen prompt sampling, changes between runs.

## Configuration

Search behaviour is controlled by environment variables read in
`src/phase4_refine.py` (defaults match the reported runs):

| Variable       | Default | Meaning                                 |
|----------------|---------|-----------------------------------------|
| `P4_SEED`      | 1234    | Qwen sampling seed (the optimiser seed) |
| `P4_TAG`       | (empty) | suffix appended to all output filenames |
| `P4_ITERS`     | 12      | refinement iterations per branch        |
| `P4_PROPOSALS` | 5       | distinct prompts kept per iteration      |

## Target images

The scripts read the target images from `TARGET_DIR`, defined near the top of
`src/vlm_caption.py`, `src/phase1_interrogate.py`, and
`src/phase3_render_score.py` (default:
`statement/TP2-students/students/tp2-chosen`). Place the target images there, or
edit `TARGET_DIR` to point at their location.

## Outputs

Each run writes two kinds of artefacts: small metrics/prompt files under `logs/`
and rendered images under `outputs/`. Everything is namespaced by the optimiser
seed via the `_s<seed>` suffix.

Metrics and prompts (`logs/`):

```
logs/
├── phase4_s<seed>_top3.csv        submitted top-3 prompts per target, with
│                                  CLIP, LPIPS, and pixel MSE/RMSE values
├── phase4_s<seed>_top3.json       the same top-3, in JSON
├── phase4_s<seed>_summary.json    per-seed summary metrics
├── phase4_s<seed>_results.json    full per-candidate results
├── phase4_s<seed>_trace.jsonl     the refinement trace (one row per step)
├── phase4_s<seed>.log             run log
└── phase4_seed_aggregate.json     metrics aggregated across all seeds
```

Rendered images (`outputs/`):

```
outputs/
├── phase4_s<seed>/                all rendered candidates for the seed (bulky)
└── phase4_s<seed>_top3/           only the submitted top-3 renders per target,
                                   named <image>_<id>_rank<n>_<metric>_cand<k>.png
```

The bulky per-candidate images under `outputs/phase4_s<seed>/` (and the
`outputs/phase3/` intermediates) are git-ignored; only the small `_top3/` renders
and the `logs/` metrics are kept in the repository.

## Delivery

`delivery/` is the self-contained submission: the comment-stripped pipeline in
`delivery/src/`, its pinned `requirements.txt`, and the final top-3 prompts and
renders per seed under `delivery/top3/`. It has its own `README.md` with
standalone run instructions. Rebuild the stripped copy from the working sources
with `python analysis/strip_and_deliver.py`. The full compiled write-up is in
`report/main.pdf`.

## Authors

Developed as a group project for the Generative AI course.
