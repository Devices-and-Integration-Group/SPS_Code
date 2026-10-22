"""Compatibility wrapper: repeat one branch with the shared 70-epoch baseline."""
import argparse
import subprocess
import sys
from pathlib import Path

from run_ablation import DEFAULT_SEEDS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "octmnist.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "multiseed")
    parser.add_argument("--experiment", choices=("float", "ideal", "measured"), default="measured")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--warmup-epochs", type=int, default=70)
    parser.add_argument("--device-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--reuse-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = [
        sys.executable, str(ROOT / "run_ablation.py"),
        "--dataset", str(args.dataset), "--output-dir", str(args.output_dir),
        "--experiments", args.experiment, "--seeds", *(str(x) for x in args.seeds),
        "--warmup-epochs", str(args.warmup_epochs),
        "--device-epochs", str(args.device_epochs),
        "--batch-size", str(args.batch_size), "--workers", str(args.workers),
        "--nproc-per-node", str(args.nproc_per_node),
    ]
    if args.reuse_completed:
        command.append("--reuse-completed")
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
