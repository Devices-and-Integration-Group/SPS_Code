import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from main import validation_keep_indices


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=root / "data" / "octmnist.npz")
    parser.add_argument("--excluded-output", type=Path, default=root / "data" / "validation_excluded_indices.csv")
    parser.add_argument("--manifest-output", type=Path, default=root / "data" / "split_indices.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for output in (args.excluded_output, args.manifest_output):
        if output.exists() and not args.overwrite:
            raise FileExistsError(output)
    with np.load(args.dataset, allow_pickle=False) as archive:
        train_images = archive["train_images"]
        train_labels = archive["train_labels"].reshape(-1)
        val_images = archive["val_images"]
        val_labels = archive["val_labels"].reshape(-1)
        test_images = archive["test_images"]
        test_labels = archive["test_labels"].reshape(-1)
        keep = validation_keep_indices(train_images, val_images)
    excluded = np.setdiff1d(np.arange(len(val_images)), keep)
    excluded_set = set(excluded.tolist())
    with args.excluded_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["official_val_index"])
        writer.writerows((int(i),) for i in excluded)
    with args.manifest_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "official_index", "label", "included", "pixel_sha256"])
        for split, images, labels in (
            ("train", train_images, train_labels),
            ("validation", val_images, val_labels),
            ("test", test_images, test_labels),
        ):
            for index, (image, label) in enumerate(zip(images, labels)):
                included = not (split == "validation" and index in excluded_set)
                writer.writerow([
                    split, index, int(label), int(included),
                    hashlib.sha256(image.tobytes()).hexdigest(),
                ])
    print(f"Excluded {len(excluded)} of {len(val_images)} validation rows -> {args.excluded_output}")
    print(f"Exact split manifest -> {args.manifest_output}")


if __name__ == "__main__":
    main()
