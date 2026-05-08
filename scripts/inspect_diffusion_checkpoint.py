import argparse
import sys
from typing import Any, Dict, List, Tuple

try:
    import torch
except Exception as exc:
    print(f"Failed to import torch: {exc}")
    print("Activate your environment and install torch first.")
    sys.exit(1)

TARGET_KEYS = [
    "learning_rate",
    "base_channels",
    "use_attention",
    "num_res_blocks",
]


def walk(obj: Any, path: str = "root") -> List[Tuple[str, Any, str]]:
    """Return all matching key paths as (key, value, path)."""
    found: List[Tuple[str, Any, str]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            next_path = f"{path}.{k}"
            if k in TARGET_KEYS:
                found.append((k, v, next_path))
            found.extend(walk(v, next_path))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(walk(v, f"{path}[{i}]"))

    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch checkpoint for selected hyperparameters."
    )
    parser.add_argument(
        "--ckpt",
        default="src/results/diffusion_generated_samples_checkpoint.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all matches for each key (not just first preference)",
    )
    args = parser.parse_args()

    try:
        obj = torch.load(args.ckpt, map_location="cpu")
    except Exception as exc:
        print(f"Failed to load checkpoint '{args.ckpt}': {exc}")
        return 1

    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Root object type: {type(obj).__name__}")

    top_level: Dict[str, Any] = obj if isinstance(obj, dict) else {}
    all_found = walk(obj)

    by_key: Dict[str, List[Tuple[Any, str]]] = {k: [] for k in TARGET_KEYS}
    for key, value, path in all_found:
        by_key[key].append((value, path))

    print("\nRequested values:")
    for key in TARGET_KEYS:
        if key in top_level:
            print(f"- {key}: {top_level[key]} (from root.{key})")
            if args.all and by_key[key]:
                for value, path in by_key[key]:
                    if path != f"root.{key}":
                        print(f"  additional match -> {path} = {value}")
            continue

        matches = by_key[key]
        if not matches:
            print(f"- {key}: <not found>")
        elif args.all:
            first_value, first_path = matches[0]
            print(f"- {key}: {first_value} (first match at {first_path})")
            for value, path in matches[1:]:
                print(f"  additional match -> {path} = {value}")
        else:
            value, path = matches[0]
            print(f"- {key}: {value} (from {path})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
