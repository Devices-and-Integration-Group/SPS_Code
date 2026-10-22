"""One-command shared-baseline OCTMNIST ablation (5 seeds x 3 weight modes)."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("float", "ideal", "measured")
DEFAULT_SEEDS = (19, 29, 39, 49, 59)
OFFICIAL_OCTMNIST_MD5 = "c68d92d5b585d8d81f7112f81e2d0842"
METRICS = ("acc", "macro_f1", "min_recall", "loss",
           "recall_CNV", "recall_DME", "recall_Drusen", "recall_Normal")


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preflight(args) -> dict:
    dataset = args.dataset.resolve()
    curve = (ROOT / "line_LTP_LTD.csv").resolve()
    required = [dataset, curve, ROOT / "data" / "split_indices.csv",
                ROOT / "data" / "validation_excluded_indices.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required data/manifest files: " + ", ".join(missing))
    dataset_md5 = digest(dataset, "md5")
    if dataset_md5.lower() != OFFICIAL_OCTMNIST_MD5:
        raise ValueError(f"OCTMNIST MD5 mismatch: {dataset_md5}; expected {OFFICIAL_OCTMNIST_MD5}")
    if args.warmup_epochs != 70 or args.device_epochs != 30:
        raise ValueError("This ablation requires exactly 70 + 30 epochs")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Random seeds must be distinct")
    if args.nproc_per_node > 1 and torch.cuda.device_count() < args.nproc_per_node:
        raise RuntimeError("--nproc-per-node exceeds the available CUDA GPU count")
    if not torch.cuda.is_available():
        print("[warning] CUDA is unavailable; the full experiment will run very slowly on CPU", flush=True)
    return {"dataset_md5": dataset_md5, "curve_sha256": digest(curve),
            "split_indices_sha256": digest(ROOT / "data" / "split_indices.csv"),
            "validation_exclusions_sha256": digest(ROOT / "data" / "validation_excluded_indices.csv"),
            "main_sha256": digest(ROOT / "main.py"),
            "runner_sha256": digest(ROOT / "run_ablation.py"),
            "seeds": args.seeds, "warmup_epochs": 70, "device_epochs": 30,
            "batch_size": args.batch_size, "workers": args.workers,
            "nproc_per_node": args.nproc_per_node,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__, "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def command(args, stage: str, seed: int, output_dir: Path, baseline_ckpt: Path | None = None) -> list[str]:
    main_args = [
        str(ROOT / "main.py"), "--dataset", str(args.dataset.resolve()),
        "--output-dir", str(output_dir.resolve()), "--experiment", stage,
        "--seed", str(seed), "--warmup-epochs", "70", "--device-epochs", "30",
        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--checkpoint", "best.pt", "--no-plots",
    ]
    if baseline_ckpt is not None:
        main_args.extend(["--base-checkpoint", str(baseline_ckpt.resolve())])
    if args.nproc_per_node > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={args.nproc_per_node}"] + main_args
    return [sys.executable] + main_args


def run_one(args, stage: str, seed: int, output_dir: Path, fingerprint: dict,
            baseline_ckpt: Path | None = None) -> None:
    metrics = output_dir / "paper_final_split_metrics.csv"
    checkpoint = output_dir / ("base_epoch_70.pt" if stage == "baseline" else "best.pt")
    marker = output_dir / "run_manifest.json"
    expected = {**fingerprint, "stage": stage, "seed": seed}
    if args.reuse_completed and metrics.is_file() and checkpoint.is_file() and marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) == expected:
            print(f"[reuse] {stage} seed {seed}: {output_dir}", flush=True)
            return
        raise RuntimeError(f"Existing run has different data or settings: {output_dir}")
    if metrics.exists() and not args.reuse_completed:
        raise FileExistsError(f"Completed output exists; use --reuse-completed or a fresh --output-dir: {metrics}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = command(args, stage, seed, output_dir, baseline_ckpt)
    print(f"[run] {stage} seed {seed}: {output_dir}", flush=True)
    with (output_dir / "run.log").open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + subprocess.list2cmdline(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not metrics.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"Run ended without required metrics/checkpoint: {output_dir}")
    marker.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_split(path: Path, split: str) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return next(row for row in rows if row["split"] == split)


def write_summaries(records: list[dict], output_dir: Path, experiments: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "seed", *METRICS]
    with (output_dir / "ablation_per_seed.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
    with (output_dir / "ablation_mean_std.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "metric", "n_seeds", "mean", "std_sample", "mean_plus_minus_std"])
        for experiment in experiments:
            subset = [row for row in records if row["experiment"] == experiment]
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
                mean = float(values.mean())
                sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                writer.writerow([experiment, metric, len(values), mean, sd,
                                 f"{mean:.6f} ± {sd:.6f}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "octmnist.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ablation")
    parser.add_argument("--experiments", choices=EXPERIMENTS, nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--warmup-epochs", type=int, default=70)
    parser.add_argument("--device-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--reuse-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate files and print commands without training")
    args = parser.parse_args()
    if args.nproc_per_node < 1 or args.batch_size < 1:
        parser.error("GPU process count and global batch size must be positive")
    fingerprint = preflight(args)
    if args.dry_run:
        for seed in args.seeds:
            base_dir = args.output_dir / "baseline" / f"seed_{seed}"
            print(subprocess.list2cmdline(command(args, "baseline", seed, base_dir)))
            for experiment in args.experiments:
                branch_dir = args.output_dir / experiment / f"seed_{seed}"
                print(subprocess.list2cmdline(command(args, experiment, seed, branch_dir,
                                                  base_dir / "base_epoch_70.pt")))
        return
    records = []
    for seed in args.seeds:
        base_dir = args.output_dir / "baseline" / f"seed_{seed}"
        run_one(args, "baseline", seed, base_dir, fingerprint)
        base_ckpt = base_dir / "base_epoch_70.pt"
        for experiment in args.experiments:
            branch_dir = args.output_dir / experiment / f"seed_{seed}"
            run_one(args, experiment, seed, branch_dir, fingerprint, base_ckpt)
            row = read_split(branch_dir / "paper_final_split_metrics.csv", "test")
            records.append({"experiment": experiment, "seed": seed,
                            **{key: row[key] for key in METRICS}})
    write_summaries(records, args.output_dir, args.experiments)
    print(f"Completed {len(records)} branch runs; summary -> {args.output_dir / 'ablation_mean_std.csv'}")


if __name__ == "__main__":
    main()
