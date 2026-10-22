# SPS OCTMNIST SNN

Code and data indices for a five-seed OCTMNIST spiking neural network (SNN) study comparing floating-point, ideal-quantized, and measured-conductance-mapped synaptic weights.

## Data and splits

Use the 28 × 28 `octmnist.npz` archive from MedMNIST v2, data release v3.0 ([Zenodo 10519652](https://zenodo.org/records/10519652), CC BY 4.0). Expected MD5: `c68d92d5b585d8d81f7112f81e2d0842`. Save it as `data/octmnist.npz`.

```bash
mkdir -p data
curl -L "https://zenodo.org/records/10519652/files/octmnist.npz?download=1" -o data/octmnist.npz
echo "c68d92d5b585d8d81f7112f81e2d0842  data/octmnist.npz" | md5sum -c -
```

On Windows, verify with `(Get-FileHash .\data\octmnist.npz -Algorithm MD5).Hash`. The loader also checks the split sizes, image dimensions, and label range.

| Split | Included / official | CNV | DME | Drusen | Normal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 97,477 / 97,477 | 33,484 | 10,213 | 7,754 | 46,026 |
| Validation | 9,483 / 10,832 | 2,759 | 1,054 | 732 | 4,938 |
| Test | 1,000 / 1,000 | 250 | 250 | 250 | 250 |

`data/split_indices.csv` gives every zero-based official NPZ row index, label, inclusion flag, and raw-pixel SHA-256 hash. `data/validation_excluded_indices.csv` lists the 1,349 validation rows removed because their raw 28 × 28 arrays exactly match a training image. Train and test retain their official indices and order; within-split duplicates are retained. The raw-pixel hash audit found no exact cross-split duplicates after filtering. The archive has no patient identifiers, so patient-level overlap cannot be determined from these files. Regenerate the derived indices with `python export_validation_exclusions.py --dataset data/octmnist.npz --overwrite`; this does not modify the NPZ.

There is no class-balanced subsampling, oversampling, or replacement sampling. Each included training image is used once per epoch. Class imbalance is handled only by loss weights `w_c = (1/sqrt(n_c)) / mean_k(1/sqrt(n_k))`, equal to `0.69666175, 1.26143235, 1.44769719, 0.59420871` for CNV, DME, Drusen, and Normal. Test data are not used for selection or tuning.

## Input and model

Raw uint8 images are divided by 255. The two input channels are the raw image and an enhanced image, where `z = clip((x-mean(x))/(2.45*max(std(x),0.08))+0.5,0,1)` followed by `clip(z+0.28*(z-smooth3x3(z)),0,1)`; `smooth3x3` is a 3 × 3 average filter. Training augmentation: horizontal flip (`p=0.5`); shared-batch translation of −1, 0, or +1 pixel per axis (`p=0.6`, zero fill); per-image contrast `[0.92,1.08]`; brightness `[−0.02,0.02]`; and Gaussian noise (`SD=0.006`, `p=0.25`). Validation and test use no augmentation or flip averaging.

For normalized intensity `x`, phase encoding uses `p = clip(1-exp(-r_in*dt*x),0,0.97)`, `r_in=9.0e5 Hz`, `dt=1 µs`, and an initial phase of `0.5`. Each step emits a spike when phase reaches one and then subtracts one. All model runs use 16 steps (16 µs).

The LIF rule is `u_t=βu_(t−1)+I_t`, `s_t=H(u_t−θ)`, followed by `u_t←u_t−s_tθ`. With measured `τ=8.7 µs`, `β=exp(−dt/τ)=exp(−1/8.7)=0.891417373`. Reset is subtractive, not a clamp to zero. The eight LIF thresholds are `0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.05`.

Architecture: two input channels → Conv48-LIF × 2 → average pool → Conv96-LIF × 2 → average pool → Conv192-LIF × 3 → average pool → FC384-LIF → FC4. Convolutions use 3 × 3 kernels, padding one, and batch normalization; FC384 uses layer normalization. Same-width convolutional residual spike-current coefficients are `0.20, 0.20, 0.15, 0.15`. The four output neurons integrate leaky temporal evidence without an output spike threshold.

## Training and weight mapping

| Setting | Value |
| --- | --- |
| Optimizer | AdamW; cosine learning-rate schedules |
| Learning rate | `8.0e-4` for the shared baseline; `2.0e-4` for every continuation |
| Weight decay | `1.5e-4` in baseline; half for non-conductance continuation parameters; zero for differential conductance |
| Batch size | 96 global for training; 384 for evaluation |
| Training | 70 shared continuous-weight epochs, then 30 epochs per branch (100 effective epochs) |
| Regularization | Label smoothing `0.035`; gradient clip `3.0`; spike penalty `2.5e-4` outside firing-rate range `[0.035,0.36]` |
| Seeds | `19, 29, 39, 49, 59` |

All branches start from the same seed-matched final baseline checkpoint, `base_epoch_70.pt`. Each uses a fresh AdamW optimizer and the same continuation schedule. `float` continues continuous-weight training; `ideal` and `measured` use straight-through-estimator quantization-aware training (QAT). The best checkpoint among epochs 71–100 maximizes `0.66*validation accuracy + 0.19*validation macro-F1 + 0.15*validation minimum class recall`. Test metrics are computed after checkpoint selection.

`line_LTP_LTD.csv` contains 80 measured LTP and 80 measured LTD conductances in siemens (`3.318e-5`–`4.19208e-4 S`). Its first data row (`Number=1`) is the zero-pulse state (`k=0`); the last is `k=79`. A signed synapse uses `ΔG=G+−G−`, `D=G_max−G_min`, and layer coefficient `w=sqrt(6/fan_in)*ΔG/D`. Trainable shadow `ΔG` is projected to `[−D,D]`.

The ideal mapper uses 80 uniformly spaced, reversible states `G_k=G_min+kD/79`, giving 159 signed differential levels. The measured mapper preserves LTP/LTD branch and pulse index, and selects the nearest target difference from `160²=25,600` ordered measured-state pairs. Quantized forward passes use the selected difference; gradients use the straight-through estimator. Mapped inference uses the selected conductances. Biases, normalization, thresholds, and other digital parameters remain digital.

## Run and outputs

Install a CUDA-enabled PyTorch build suitable for the target GPU; see `requirements.txt`. Run all five seeds and three branches with:

```bash
python run_ablation.py --dataset data/octmnist.npz --output-dir results/ablation
```

Use `python run_ablation.py --dataset data/octmnist.npz --dry-run` to check inputs and commands first. A direct `main.py` run trains only one 70-epoch baseline; continuations require its seed-matched `base_epoch_70.pt` via `--base-checkpoint`. On two GPUs, add `--nproc-per-node 2`; batch size 96 remains global. Use `--reuse-completed` only for matching completed runs, or choose a fresh output directory.

`results/ablation/` contains each seed's baseline and `float`, `ideal`, and `measured` continuations: `run_manifest.json`, `run.log`, training histories, firing-rate tables, split metrics, confusion matrices, and checkpoints. `best.pt` is validation-selected; the measured branch also has `programmed.pt` for selected `G+/G−` states. Summary tables are `ablation_per_seed.csv` and `ablation_mean_std.csv`. To regenerate Fig. 5c–g source data and plots, run:

```bash
python reproduce_fig5_revised.py --dataset data/octmnist.npz --results-dir results/ablation --output-dir results/figure5_revised
```

Use an empty figure output directory.

## Completed five-seed test results

Final validation-selected checkpoint metrics on the official 1,000-image test split; mean ± sample standard deviation (`ddof=1`):

| Weight condition | Accuracy | Macro-F1 | Drusen recall |
| --- | ---: | ---: | ---: |
| Floating-point | 76.40 ± 0.35% | 73.43 ± 0.90% | 29.76 ± 4.17% |
| Ideal quantization | 76.54 ± 1.38% | 73.61 ± 1.98% | 30.32 ± 4.74% |
| Measured mapping | 76.86 ± 0.77% | 74.09 ± 1.34% | 31.76 ± 4.36% |

Training environment: Python `3.12.12`, NumPy `2.3.5`, PyTorch `2.8.0+cu128` (CUDA runtime `12.8`), NVIDIA GeForce RTX 4090.
