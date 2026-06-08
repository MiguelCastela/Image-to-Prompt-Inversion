# TP2: Image-to-Prompt Inversion

Source code to generate, evaluate, and select recovered text prompts for a
fixed text-to-image generator (`SimianLuo/LCM_Dreamshaper_v7`). Given a target
image, the pipeline produces a warm-start prompt, then refines it in a closed
loop that renders each candidate, scores it against the target, and keeps the
best prompts.

## Dependencies

A CUDA-capable GPU and Python 3.10 or newer are required. Install the pinned
dependencies with:

```
pip install -r requirements.txt
```

Model checkpoints (the LCM generator, BLIP-2, the CLIP encoders, and the
Qwen2.5-VL refiner) download automatically on first run. All models are
open-weights and run locally; no API key is needed for the submission pipeline.

## Folder structure

```
delivery/
  README.md
  requirements.txt
  src/                          source code (comments removed)
    vlm_caption.py              Stage 1: BLIP-2 subject and caption
    phase1_interrogate.py       Stage 1: CLIP-Interrogator warm-start prompt
    phase2_sampling.py          shared utilities (Qwen loading, parsing)
    phase3_render_score.py      shared utilities (LCM rendering, ranking, summary)
    phase4_refine.py            Stage 2: closed-loop refinement and selection
    evaluation.py               metrics (CLIP, LPIPS, pixel MSE/RMSE)
    phase4_aggregate_seeds.py   aggregate metrics across optimiser seeds
  top3/                         submitted top-3 per seed (4 completed seeds)
    seed_1234/  top3.csv, top3.json, 18 rendered images
    seed_3456/  ...
    seed_5678/  ...
    seed_9012/  ...
```

`phase4_refine.py` imports helpers from `phase2_sampling.py` and
`phase3_render_score.py`, so all files in `src/` must stay in the same
directory. Run every command from inside `src/`.

## Target images

The scripts read the six target images from the directory `TARGET_DIR`, defined
near the top of `vlm_caption.py`, `phase1_interrogate.py`, and
`phase3_render_score.py` (default:
`statement/TP2-students/students/tp2-chosen`). Place the target images there, or
edit `TARGET_DIR` in those files to point at their location.

## Running the pipeline

Run the four steps in order from `src/`. Intermediate results are written as
JSON files in the working directory, and rendered images under `outputs/`.

1. Stage 1, subject caption (BLIP-2):
   ```
   python vlm_caption.py
   ```
   Writes `phase1_captions.json`.

2. Stage 1, warm-start prompt (CLIP-Interrogator plus the subject):
   ```
   python phase1_interrogate.py
   ```
   Writes `phase1_warmstart.json`.

3. Stage 2, closed-loop refinement (generate, render, score, select):
   ```
   P4_SEED=1234 P4_TAG=_s1234 python phase4_refine.py
   ```
   Writes `phase4_s1234_results.json` (every scored candidate),
   `phase4_s1234_top3.json` and `phase4_s1234_top3.csv` (the submitted top-3 per
   target), `phase4_s1234_branches.json`, and `phase4_s1234_summary.json`.

   For the multi-seed protocol, repeat step 3 with each seed, changing both
   variables together:
   ```
   P4_SEED=3456 P4_TAG=_s3456 python phase4_refine.py
   P4_SEED=5678 P4_TAG=_s5678 python phase4_refine.py
   P4_SEED=9012 P4_TAG=_s9012 python phase4_refine.py
   P4_SEED=7890 P4_TAG=_s7890 python phase4_refine.py
   ```
   The LCM render seed stays fixed per image (the leading digits of each
   filename); only `P4_SEED`, which controls the Qwen prompt sampling, changes
   between runs.

4. Aggregate across seeds (mean and standard deviation of each metric):
   ```
   python phase4_aggregate_seeds.py
   ```
   Reads every `phase4_s*_summary.json` and writes `phase4_seed_aggregate.json`.

## Configuration

The generator and search settings are environment variables read in
`phase4_refine.py` (defaults match the reported runs):

| Variable        | Default | Meaning                                  |
|-----------------|---------|------------------------------------------|
| `P4_SEED`       | 1234    | Qwen sampling seed (the optimiser seed)  |
| `P4_TAG`        | (empty) | suffix appended to all output filenames  |
| `P4_ITERS`      | 12      | refinement iterations per branch         |
| `P4_PROPOSALS`  | 5       | distinct prompts kept per iteration      |

## Outputs

For each seed, `phase4_s<seed>_top3.csv` lists the submitted top-3 prompts per
target with their CLIP, LPIPS, and pixel MSE/RMSE values, and the corresponding
rendered images are saved under `outputs/phase4_s<seed>_top3/`. The `top3/`
folder in this delivery contains these files for all five seeds (1234, 3456,
5678, 9012, 7890 and 7890).
