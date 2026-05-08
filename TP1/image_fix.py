from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def split_grid_image(image_path: Path, rows: int, cols: int, trim_px: int) -> list[list[np.ndarray]]:
	"""Split a single image containing a rows x cols grid into individual tiles."""
	arr = np.array(Image.open(image_path).convert("RGB"))
	height, width = arr.shape[:2]

	y_edges = np.rint(np.linspace(0, height, rows + 1)).astype(int)
	x_edges = np.rint(np.linspace(0, width, cols + 1)).astype(int)

	tiles: list[list[np.ndarray]] = []
	for r in range(rows):
		row_tiles = []
		for c in range(cols):
			y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
			x0, x1 = int(x_edges[c]), int(x_edges[c + 1])

			# Trim border pixels to remove separator lines from the source grid.
			if trim_px > 0:
				y0 = min(y0 + trim_px, y1 - 1)
				y1 = max(y1 - trim_px, y0 + 1)
				x0 = min(x0 + trim_px, x1 - 1)
				x1 = max(x1 - trim_px, x0 + 1)

			row_tiles.append(arr[y0:y1, x0:x1])
		tiles.append(row_tiles)

	return tiles


def save_progression_plot(
	tiles: list[list[np.ndarray]],
	epochs: list[int],
	output_path: Path,
	title: str,
) -> None:
	rows = len(tiles)
	cols = len(tiles[0]) if rows else 0
	if cols == 0:
		raise ValueError("No tiles were extracted from the source image.")
	if len(epochs) != cols:
		raise ValueError(f"Expected {cols} epoch labels, got {len(epochs)}.")

	fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
	if cols == 1:
		axes = axes.reshape(rows, 1)

	for col, epoch in enumerate(epochs):
		for row in range(rows):
			ax = axes[row, col]
			ax.imshow(tiles[row][col])
			ax.axis("off")
			if row == 0:
				ax.set_title(f"Epoch {epoch}")
			if col == 0:
				ax.set_ylabel(f"Image {row + 1}")

	fig.suptitle(title)
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def resolve_default_inputs(base_dir: Path) -> list[Path]:
	candidates = [base_dir / "image.png", base_dir / "image2.png", base_dir / "image 2.png"]
	found: list[Path] = []
	for path in candidates:
		if path.exists() and path not in found:
			found.append(path)
	return found


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Convert 3x4 grid images into VAE-style progression plots "
			"with epoch titles and row labels."
		)
	)
	parser.add_argument(
		"--inputs",
		nargs="*",
		default=None,
		help="Input image paths. If omitted, image.png and image2.png are auto-detected.",
	)
	parser.add_argument("--rows", type=int, default=3, help="Number of grid rows in each source image.")
	parser.add_argument("--cols", type=int, default=4, help="Number of grid columns in each source image.")
	parser.add_argument(
		"--epochs",
		nargs="*",
		type=int,
		default=[1, 50, 100, 200],
		help="Epoch labels for columns. Must match --cols.",
	)
	parser.add_argument(
		"--trim-px",
		type=int,
		default=1,
		help="Pixels trimmed from each tile edge to remove separators.",
	)
	parser.add_argument(
		"--title",
		type=str,
		default="Generation Evolution (fixed seed)",
		help="Figure title.",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=None,
		help="Output directory. Defaults to each input image directory.",
	)
	parser.add_argument(
		"--suffix",
		type=str,
		default="_vae_style.png",
		help="Suffix added to each output filename stem.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.rows <= 0 or args.cols <= 0:
		raise ValueError("--rows and --cols must be positive integers.")
	if args.trim_px < 0:
		raise ValueError("--trim-px must be >= 0.")
	if len(args.epochs) != args.cols:
		raise ValueError("Number of --epochs labels must match --cols.")

	if args.inputs:
		input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
	else:
		input_paths = resolve_default_inputs(Path.cwd())

	if not input_paths:
		raise FileNotFoundError("No input images found. Pass --inputs or place image.png/image2.png in the CWD.")

	for image_path in input_paths:
		if not image_path.exists():
			raise FileNotFoundError(f"Input image not found: {image_path}")

		tiles = split_grid_image(image_path, rows=args.rows, cols=args.cols, trim_px=args.trim_px)
		out_dir = args.out_dir if args.out_dir is not None else image_path.parent
		output_path = out_dir / f"{image_path.stem}{args.suffix}"
		save_progression_plot(tiles, args.epochs, output_path, args.title)
		print(f"Saved: {output_path}")


if __name__ == "__main__":
	main()
