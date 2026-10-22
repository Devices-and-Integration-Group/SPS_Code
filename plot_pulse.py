import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ("CNV", "DME", "Drusen", "Normal")
PIXEL_FRACTIONS = ((0.5, 0.5), (0.25, 0.25), (0.75, 0.75))
INPUT_RATE_HZ = 9.0e5
DT_S = 1.0e-6


def enhance_oct(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32) / np.float32(255.0)
    mean = x.mean()
    std = max(float(x.std(ddof=1)), 0.08)
    z = np.clip((x - mean) / np.float32(2.45 * std) + np.float32(0.5), 0.0, 1.0)
    height, width = z.shape
    padded = np.pad(z, 1, mode="constant")
    smooth = sum(padded[dy:dy + height, dx:dx + width]
                 for dy in range(3) for dx in range(3)) / np.float32(9.0)
    return np.clip(z + np.float32(0.28) * (z - smooth), 0.0, 1.0)


def encode_pixels(values: np.ndarray, steps: int) -> np.ndarray:
    intensity = values.astype(np.float32, copy=True)
    increment = np.clip(np.float32(1.0) - np.exp(-np.float32(INPUT_RATE_HZ * DT_S) * intensity),
                        0.0, 0.97)
    phase = np.full_like(increment, 0.5)
    raster = np.empty((len(values), steps), dtype=np.uint8)
    for step in range(steps):
        phase = phase + increment
        spikes = (phase >= 1.0).astype(np.uint8)
        phase = phase - spikes.astype(np.float32)
        raster[:, step] = spikes
    return raster


def load_samples(dataset_path: Path) -> list[dict]:
    with np.load(dataset_path, allow_pickle=False) as dataset:
        images = dataset["train_images"]
        labels = dataset["train_labels"].reshape(-1)
        samples = []
        for label, name in enumerate(CLASS_NAMES):
            matches = np.flatnonzero(labels == label)
            if matches.size == 0:
                raise ValueError(f"No training image found for {name}")
            index = int(matches[0])
            samples.append({"class": name, "index": index, "image": images[index].copy()})
    return samples


def select_pixels(samples: list[dict]) -> list[dict]:
    records = []
    for sample in samples:
        image = sample["image"]
        enhanced = enhance_oct(image)
        height, width = image.shape
        for fy, fx in PIXEL_FRACTIONS:
            y, x = int(height * fy), int(width * fx)
            records.append({
                "class": sample["class"],
                "sample_index": sample["index"],
                "pixel_y": y,
                "pixel_x": x,
                "enhanced_intensity": float(enhanced[y, x]),
            })
    return records


def save_plot(records: list[dict], raster: np.ndarray, output_dir: Path, steps: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    events = [np.flatnonzero(row) for row in raster]
    ax.eventplot(events, colors="#1a47b8", lineoffsets=[float(i) for i in range(len(records))],
                 linelengths=0.8, linewidths=0.8)
    ax.set_xlabel("Number of time steps", fontsize=20, labelpad=15)
    ax.set_xlim(0, steps)
    ax.set_ylim(-0.5, len(records) - 0.5)
    ax.set_yticks(np.arange(len(records)))
    ax.set_yticklabels([f"{item['class']}\n({item['pixel_y']},{item['pixel_x']})"
                        for item in records], fontsize=10)
    for group in range(1, len(CLASS_NAMES)):
        ax.axhline(group * len(PIXEL_FRACTIONS) - 0.5, color="gray",
                   linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "input_pulse_raster.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "input_pulse_raster.svg", bbox_inches="tight")
    plt.close(fig)


def save_data(records: list[dict], raster: np.ndarray, samples: list[dict],
              output_dir: Path, dataset_path: Path) -> None:
    np.save(output_dir / "pulse_matrix.npy", raster)
    with (output_dir / "pulse_details.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "train_index", "pixel_y", "pixel_x",
                         "enhanced_intensity", "spike_steps"))
        for item, row in zip(records, raster):
            writer.writerow((item["class"], item["sample_index"], item["pixel_y"],
                             item["pixel_x"], f"{item['enhanced_intensity']:.9g}",
                             ";".join(map(str, np.flatnonzero(row)))))
    for sample in samples:
        plt.imsave(output_dir / f"sample_{sample['class']}.png", sample["image"],
                   cmap="gray", vmin=0, vmax=255)
    metadata = {
        "dataset": str(dataset_path),
        "input_rate_hz": INPUT_RATE_HZ,
        "time_step_s": DT_S,
        "steps": int(raster.shape[1]),
        "channel": "contrast_enhanced",
        "sample_selection": "first training image of each class",
        "matrix_shape": list(raster.shape),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=SCRIPT_DIR / "data" / "octmnist.npz")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "input_pulse_raster")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    dataset_path = args.dataset.resolve()
    if not dataset_path.is_file():
        parser.error(f"Dataset not found: {dataset_path}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(f"Output directory is not empty: {output_dir}; use --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(dataset_path)
    records = select_pixels(samples)
    values = np.asarray([item["enhanced_intensity"] for item in records], dtype=np.float32)
    raster = encode_pixels(values, args.steps)
    save_plot(records, raster, output_dir, args.steps)
    save_data(records, raster, samples, output_dir, dataset_path)
    print(f"Saved {len(records)} pixel trains across {args.steps} steps to {output_dir}")


if __name__ == "__main__":
    main()
