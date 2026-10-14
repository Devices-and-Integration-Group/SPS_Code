# SPS OCTMNIST SNN

This folder contains the OCTMNIST SNN training script and the measured LTP/LTD conductance table. The loader uses every image in the original MedMNIST train, validation, and test splits. It does not balance classes or select a subset. Training batches are shuffled; validation and test images are not.

## Files

- `main.py`: training, evaluation, and figure-generation code.
- `line_LTP_LTD.csv`: measured conductance states used by the weight mapper.
- `data/octmnist.npz`: expected local dataset location; the archive is not included in this repository.

## Dataset

Download the 28 × 28 OCTMNIST archive from the [MedMNIST Zenodo record](https://zenodo.org/records/10519652/files/octmnist.npz?download=1) and save it as `data/octmnist.npz`, or pass its existing path with `--dataset`. The official archive MD5 is `c68d92d5b585d8d81f7112f81e2d0842`. The source archive used to prepare this release matches that MD5.

| Split | Images | CNV | DME | Drusen | Normal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 97,477 | 33,484 | 10,213 | 7,754 | 46,026 |
| Validation | 10,832 | 3,721 | 1,135 | 862 | 5,114 |
| Test | 1,000 | 250 | 250 | 250 | 250 |

The loader checks the expected split sizes, image dimensions, and label range before training. No images are moved across splits. The original class imbalance in train and validation is retained. Existing image augmentation and the fixed loss weights are unchanged from the supplied script.

## Run

Install the packages listed in `requirements.txt` in a suitable Python environment, then run:

```bash
python main.py --dataset path/to/octmnist.npz
```

If the archive is placed at `data/octmnist.npz`, omit `--dataset`. Use `python main.py --help` for the existing training options. This release does not include checkpoints or previously generated metrics: those results came from the earlier subsampled protocol and must not be presented as results of full-split training. The script writes new checkpoints, CSV metrics, and figures into its own directory; these outputs are ignored by Git.

## GPU training

For one GPU, use the `python main.py` command above. For three GPUs on one server with a CUDA-enabled PyTorch installation, run:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=3 main.py --batch-size 96
```

Change `--nproc-per-node` to the number of GPUs to use. `--batch-size` is the global training batch size, so 96 gives approximately 32 images per GPU with three GPUs. The multi-GPU loader uses every training image once per epoch without duplicating a padded tail; validation, checkpoint selection, and final evaluation run on rank 0. `--workers` applies to each process. Multi-GPU training needs sufficient GPU memory on every participating card, and speedup depends on the hardware and data-loading performance.

The local CPU smoke test can be run with `python tests/distributed_smoke.py`; it launches two training processes. Set `SPS_TEST_NPROCS` to `1` or `3` to test those process counts. This smoke test checks execution and parameter synchronization, not model accuracy on OCTMNIST.
