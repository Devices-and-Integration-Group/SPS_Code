import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from plot_pulse import CLASS_NAMES, encode_pixels, load_samples, select_pixels


ROOT = Path(__file__).resolve().parent
MODES = ("float", "ideal", "measured")
COLORS = {"float": "#263b53", "ideal": "#509bb5", "measured": "#c18b36"}
LAYERS = ("LIF_l11", "LIF_l12", "LIF_l21", "LIF_l22",
          "LIF_l31", "LIF_l32", "LIF_l33", "LIF_fc1")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_figure(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def panel_c(dataset, output_dir):
    samples = load_samples(dataset)
    records = select_pixels(samples)
    values = np.asarray([row["enhanced_intensity"] for row in records], dtype=np.float32)
    raster = encode_pixels(values, 16)
    rows = []
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for row_id, (record, spikes) in enumerate(zip(records, raster)):
        for step in np.flatnonzero(spikes):
            rows.append({"class": record["class"], "train_index": record["sample_index"],
                         "channel": "enhanced", "pixel_y": record["pixel_y"],
                         "pixel_x": record["pixel_x"], "time_step_zero_based": int(step),
                         "time_us": int(step) + 1})
        ax.eventplot(np.flatnonzero(spikes) + 1, lineoffsets=row_id,
                     linelengths=0.7, colors="#2c668a")
    ax.set(xlim=(0.5, 16.5), ylim=(-0.7, len(records) - 0.3),
           xlabel="Time step (1 us each)", ylabel="Selected pixel / class")
    ax.set_xticks(np.arange(1, 17))
    ax.set_yticks(np.arange(len(records)))
    ax.set_yticklabels([f"{r['class']} ({r['pixel_y']},{r['pixel_x']})" for r in records],
                       fontsize=8)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig5c_raster_16step")
    write_csv(output_dir / "fig5c_raster_events.csv",
              ("class", "train_index", "channel", "pixel_y", "pixel_x",
               "time_step_zero_based", "time_us"), rows)
    return {"sample_rule": "first official training image in each class",
            "channel": "enhanced only; model also uses raw channel",
            "pixel_rule": "three fixed fractional positions per image",
            "steps": 16, "event_count": len(rows)}


def panel_d(output_dir):
    blocks = [
        ("OCT", "28 x 28", "raw + enhanced"),
        ("Encode", "16 x 2 x 28 x 28", "phase spikes"),
        ("Block 1", "48 x 14 x 14", "Conv-LIF x 2; pool"),
        ("Block 2", "96 x 7 x 7", "Conv-LIF x 2; pool"),
        ("Block 3", "192 x 3 x 3", "Conv-LIF x 3; pool"),
        ("Readout", "384 LIF -> 4", "FC; leaky evidence"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 2.8))
    ax.set(xlim=(-0.1, 13.3), ylim=(0, 2.7))
    ax.axis("off")
    for index, (name, shape, detail) in enumerate(blocks):
        x = index * 2.23
        box = FancyBboxPatch((x, 0.62), 1.85, 1.5,
                             boxstyle="round,pad=0.06", edgecolor="#48617d",
                             facecolor="#edf2f7", linewidth=1.4)
        ax.add_patch(box)
        ax.text(x + 0.925, 1.76, name, ha="center", va="center", weight="bold")
        ax.text(x + 0.925, 1.33, shape, ha="center", va="center", fontsize=9)
        ax.text(x + 0.925, 0.9, detail, ha="center", va="center", fontsize=8)
        if index < len(blocks) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.94, 1.37), (x + 2.17, 1.37),
                                         arrowstyle="->", mutation_scale=12,
                                         color="#48617d", linewidth=1.3))
    ax.text(6.65, 0.15,
            "8 LIF layers; beta = exp(-1/8.7); subtractive reset; digital normalization and bias",
            ha="center", va="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig5d_architecture")
    write_csv(output_dir / "fig5d_architecture.csv", ("block", "shape", "operation"),
              [dict(zip(("block", "shape", "operation"), item)) for item in blocks])


def collect_history(results, seeds):
    collected = {}
    for mode in MODES:
        runs = []
        for seed in seeds:
            path = results / mode / f"seed_{seed}" / "paper_training_history_all.csv"
            rows = read_csv(path)
            epochs = [int(row["global_epoch"]) for row in rows]
            if epochs != list(range(1, 101)):
                raise ValueError(f"Expected epochs 1-100 in {path}")
            runs.append(rows)
        collected[mode] = runs
    for seed_index in range(len(seeds)):
        baseline = collected["float"][seed_index]
        for mode in MODES[1:]:
            other = collected[mode][seed_index]
            for epoch_index in range(70):
                for metric in ("train_acc", "train_loss", "val_acc", "val_loss"):
                    if not np.isclose(float(baseline[epoch_index][metric]),
                                      float(other[epoch_index][metric]), atol=1e-9):
                        raise ValueError("The shared 70-epoch history differs between branches")
    return collected


def panel_e(results, seeds, output_dir):
    histories = collect_history(results, seeds)
    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for mode in MODES:
        runs = histories[mode]
        epochs = np.arange(1, 101)
        for metric, axis, scale in (("val_acc", axes[0], 100),
                                    ("val_loss", axes[1], 1)):
            values = np.array([[float(row[metric]) for row in run] for run in runs]) * scale
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            axis.plot(epochs, mean, color=COLORS[mode], label=mode, linewidth=1.8)
            axis.fill_between(epochs, mean - sd, mean + sd, color=COLORS[mode], alpha=0.13)
        for epoch_index in range(100):
            record = {"condition": mode, "epoch": epoch_index + 1,
                      "n_seeds": len(seeds)}
            for metric in ("train_acc", "train_loss", "val_acc", "val_loss"):
                values = np.array([float(run[epoch_index][metric]) for run in runs])
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_sd_sample"] = float(values.std(ddof=1))
            summary.append(record)
    for axis in axes:
        axis.axvline(70, color="#777777", linestyle="--", linewidth=1,
                     label="branch point" if axis is axes[0] else None)
        axis.set_xlabel("Effective training epoch")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].set_ylabel("Validation loss")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig5e_validation_curves")
    fields = ("condition", "epoch", "n_seeds") + tuple(
        f"{metric}_{suffix}" for metric in ("train_acc", "train_loss", "val_acc", "val_loss")
        for suffix in ("mean", "sd_sample"))
    write_csv(output_dir / "fig5e_epoch_mean_sd.csv", fields, summary)
    return {"history_source": "per-epoch training and validation metrics",
            "error_band": "sample standard deviation over seeds"}


def panel_f(results, seed, output_dir):
    path = results / "measured" / f"seed_{seed}" / "paper_confusion_matrix_test.csv"
    rows = read_csv(path)
    if len(rows) != len(CLASS_NAMES) + 1:
        raise ValueError(f"Unexpected confusion matrix shape in {path}")
    matrix = np.array([[int(row[f"pred_{name}"]) for name in CLASS_NAMES]
                       for row in rows[:len(CLASS_NAMES)]])
    if [row["true_class"] for row in rows[:len(CLASS_NAMES)]] != list(CLASS_NAMES):
        raise ValueError(f"Unexpected class order in {path}")
    if matrix.sum() != 1000:
        raise ValueError(f"Expected 1000 test images in {path}")
    fig, ax = plt.subplots(figsize=(5.3, 4.6))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=250)
    ax.set(xticks=np.arange(4), yticks=np.arange(4), xlabel="Predicted class",
           ylabel="True class", title=f"Measured mapping; seed {seed}; n=1000")
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > 125 else "#263b53")
    fig.tight_layout()
    save_figure(fig, output_dir, f"fig5f_test_confusion_seed_{seed}")
    write_csv(output_dir / f"fig5f_test_confusion_seed_{seed}.csv",
              ("true_class",) + tuple(f"pred_{name}" for name in CLASS_NAMES),
              [{"true_class": name, **{f"pred_{label}": int(matrix[i, j])
                                      for j, label in enumerate(CLASS_NAMES)}}
               for i, name in enumerate(CLASS_NAMES)])
    return {"seed": seed, "n_test_images": int(matrix.sum()),
            "accuracy": float(np.trace(matrix) / matrix.sum())}


def panel_g(results, seeds, output_dir):
    values = np.empty((len(seeds), len(LAYERS)))
    for seed_index, seed in enumerate(seeds):
        path = results / "measured" / f"seed_{seed}" / "paper_lif_firing_rates_full_training.csv"
        rows = read_csv(path)
        by_layer = {row["lif_layer"]: row for row in rows}
        if set(by_layer) != set(LAYERS):
            raise ValueError(f"Unexpected LIF layers in {path}")
        for layer_index, layer in enumerate(LAYERS):
            row = by_layer[layer]
            if int(row["recorded_epochs"]) != 100:
                raise ValueError(f"Expected 100 recorded epochs in {path}")
            values[seed_index, layer_index] = float(row["full_training_mean_firing_rate"])
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(1, len(LAYERS) + 1)
    ax.errorbar(x, mean, yerr=sd, fmt="o", capsize=4, color=COLORS["measured"])
    ax.set(xlabel="LIF layer", ylabel="Epoch-mean firing rate",
           ylim=(0, max(0.2, float((mean + sd).max()) * 1.2)))
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig5g_firing_rates_mean_sd")
    write_csv(output_dir / "fig5g_firing_rates_mean_sd.csv",
              ("lif_layer", "n_seeds", "mean", "sd_sample", "definition"),
              [{"lif_layer": layer, "n_seeds": len(seeds), "mean": float(mean[i]),
                "sd_sample": float(sd[i]),
                "definition": "mean of 100 epoch-averaged firing rates"}
               for i, layer in enumerate(LAYERS)])
    return {"definition": "100-epoch average of per-layer mean firing rates",
            "error_bar": "sample standard deviation over seeds"}


def main():
    parser = argparse.ArgumentParser(description="Create Fig. 5c-g sources from completed runs")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "octmnist.npz")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "ablation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figure5_revised")
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 29, 39, 49, 59])
    parser.add_argument("--representative-seed", type=int, default=19)
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("At least two distinct seeds are required for sample standard deviation")
    if args.representative_seed not in args.seeds:
        parser.error("--representative-seed must be one of --seeds")
    dataset = args.dataset.resolve()
    results = args.results_dir.resolve()
    output = args.output_dir.resolve()
    if not dataset.is_file():
        parser.error(f"Dataset not found: {dataset}")
    if not results.is_dir():
        parser.error(f"Results directory not found: {results}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}; choose a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"source_dataset_md5": hashlib.md5(dataset.read_bytes()).hexdigest(),
                "seeds": args.seeds, "results_directory_name": results.name,
                "plot_runtime": {"python": sys.version.split()[0],
                                 "numpy": np.__version__,
                                 "matplotlib": matplotlib.__version__},
                "panels": {}}
    if metadata["source_dataset_md5"] != "c68d92d5b585d8d81f7112f81e2d0842":
        parser.error("Dataset MD5 does not match the recorded OCTMNIST archive")
    metadata["panels"]["c"] = panel_c(dataset, output)
    panel_d(output)
    metadata["panels"]["d"] = {"definition": "schematic redrawn from main.py architecture"}
    metadata["panels"]["e"] = panel_e(results, args.seeds, output)
    metadata["panels"]["f"] = panel_f(results, args.representative_seed, output)
    metadata["panels"]["g"] = panel_g(results, args.seeds, output)
    source_files = [ROOT / "main.py", ROOT / "plot_pulse.py",
                    ROOT / "line_LTP_LTD.csv"]
    for mode in MODES:
        for seed in args.seeds:
            source_files.append(results / mode / f"seed_{seed}" / "paper_training_history_all.csv")
    for seed in args.seeds:
        source_files.append(results / "measured" / f"seed_{seed}" /
                            "paper_lif_firing_rates_full_training.csv")
    source_files.append(results / "measured" / f"seed_{args.representative_seed}" /
                        "paper_confusion_matrix_test.csv")
    metadata["source_sha256"] = {
        (str(path.relative_to(results)) if path.is_relative_to(results) else path.name): digest(path)
        for path in source_files
    }
    (output / "figure5_revised_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Created Fig. 5c-g source tables and plots in {output}")


if __name__ == "__main__":
    main()
