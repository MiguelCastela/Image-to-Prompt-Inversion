"""
Aggregate Phase-4 metrics across optimiser-seed repetitions.

The statement's Reporting Protocol requires, for a stochastic prompt search,
repeating with >= 5 seeds and reporting the mean and standard deviation of the
main metrics across those repetitions (the LCM render seed stays fixed). Each
repetition writes its own phase4_s<seed>_summary.json; this script reads all of
them and reports, for each metric, the mean +/- std of the per-run aggregate.

CPU-only (pure JSON): no model, no GPU. Run after run_phase4_sweep.sh.

Usage:
    python phase4_aggregate_seeds.py                 # globs phase4_s*_summary.json
    python phase4_aggregate_seeds.py a.json b.json   # explicit files
"""
import glob
import json
import statistics
import sys
from pathlib import Path

METRICS = ["clip_similarity", "lpips", "pixel_mse", "pixel_rmse"]
# Which aggregate block to summarise across runs. best_per_image_top1 is the
# headline (the single best prompt per image); submitted_top3 covers all
# delivered prompts.
BLOCKS = ["best_per_image_top1", "submitted_top3"]


def collect(files: list[str]) -> dict:
    runs = []
    for f in files:
        d = json.loads(Path(f).read_text())
        runs.append({
            "file": f,
            "gen_seed": d.get("metadata", {}).get("gen_seed"),
            "aggregate": d.get("aggregate", {}),
        })
    return runs


def main() -> None:
    args = sys.argv[1:]
    files = sorted(args) if args else sorted(glob.glob("phase4_s*_summary.json"))
    if not files:
        sys.exit("No per-seed summaries found (phase4_s*_summary.json). "
                 "Run run_phase4_sweep.sh first.")

    runs = collect(files)
    seeds = [r["gen_seed"] for r in runs]
    print(f"Aggregating {len(runs)} repetition(s); optimiser seeds: {seeds}")
    print("(LCM render seed stays fixed per image — only the prompt search varied.)\n")

    out = {"n_repetitions": len(runs), "optimiser_seeds": seeds,
           "source_files": files, "across_seeds": {}}

    for block in BLOCKS:
        print(f"== {block} : mean +/- std ACROSS {len(runs)} repetitions ==")
        out["across_seeds"][block] = {}
        for m in METRICS:
            vals = [r["aggregate"].get(block, {}).get(m, {}).get("mean")
                    for r in runs]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            out["across_seeds"][block][m] = {
                "mean": round(mean, 6), "std": round(std, 6),
                "per_run": [round(v, 6) for v in vals], "n": len(vals),
            }
            arrow = "(higher better)" if m == "clip_similarity" else "(lower better)"
            print(f"  {m:<16} {mean:.4f} +/- {std:.4f}  {arrow}")
        print()

    Path("phase4_seed_aggregate.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("Saved -> phase4_seed_aggregate.json")


if __name__ == "__main__":
    main()
