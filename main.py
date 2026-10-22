"""
main.py
=======
Shared 70-epoch continuous-weight baseline and reusable OCTMNIST SNN components.

Key corrections in this version
-------------------------------
1) The trainable ideal-SNN synaptic variable is the differential conductance DeltaG (siemens),
   not an unconstrained abstract weight.  Every optimizer step projects DeltaG into the physically
   achievable range implied by the measured LTP/LTD file.
2) Float -> device mapping is NOT a linear pulse-index mapping.  The code enumerates all measured
   LTP and LTD conductance states for a differential pair (G+ - G-) and quantizes each trained
   DeltaG to the nearest actually measured discrete differential-conductance state.
3) QAT forward passes use those measured discrete weights with a straight-through estimator.
   Final programmed inference uses the same selected measured G+ and G- states, so QAT and
   programmed-device weights are numerically identical apart from floating-point roundoff.
4) One SNN time window is used everywhere: exactly 16 steps for training, validation, testing,
   Grad-CAM and smoke tests.  With dt=1 us the algorithm/physical window is 16 us.
5) The network is strengthened to 48-48 / 96-96 / 192-192-192 / FC384, uses raw+enhanced OCT
   as two spike-encoded input channels, residual spike-current shortcuts inside equal-width blocks,
   and a bounded monotonic output transform.  The latter leaves argmax unchanged while keeping
   four-class cross-entropy below 10 by construction.
6) This entry point trains only the 70-epoch continuous device-range-constrained baseline.
   run_ablation.py branches from its final checkpoint for 30 further epochs per condition.
7) CSV export of epoch metrics, confusion matrices and eight LIF-layer firing rates is retained.

The code targets >=90% validation/test accuracy, but it never uses test labels to manufacture that
number.  If the data/training run does not reach the target, it reports NOT REACHED truthfully.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

SCRIPT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ("CNV", "DME", "Drusen", "Normal")
N_CLASSES = 4
TAU_LEAK_S = 8.7e-6

# CSV export labels for the eight LIF layers.  This is recording-only metadata;
# it does not participate in forward propagation or training.
LIF_LAYER_NAMES = ("LIF_l11", "LIF_l12", "LIF_l21", "LIF_l22",
                   "LIF_l31", "LIF_l32", "LIF_l33", "LIF_fc1")

# =============================================================================
# Reproducibility / files
# =============================================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def distributed_context(local_rank_arg: Optional[int] = None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device
    local_rank = int(os.environ.get("LOCAL_RANK", local_rank_arg if local_rank_arg is not None else 0))
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK={local_rank} exceeds available CUDA devices")
        if not dist.is_nccl_available():
            raise RuntimeError("This PyTorch build does not support NCCL")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(minutes=30))
    return dist.get_rank(), dist.get_world_size(), device

def beta_from_dt(dt_s: float) -> float:
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")
    b = math.exp(-float(dt_s) / TAU_LEAK_S)
    if not (0.0 < b < 1.0):
        raise ValueError(f"LIF beta must be in (0,1), got {b}")
    return b

def find_input_file(canonical: str, patterns: list[str]) -> Path:
    p = SCRIPT_DIR / canonical
    if p.exists():
        return p
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(sorted(SCRIPT_DIR.glob(pat)))
    hits = [h for h in hits if h.is_file()]
    if not hits:
        raise FileNotFoundError(f"Cannot find {canonical} next to {Path(__file__).name}")
    print(f"[file] {canonical} not found; using {hits[0].name}")
    return hits[0]

# =============================================================================
# Measured memristor LTP/LTD: strictly sequential pulse-state programming
# =============================================================================
def read_device_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8-sig")
    names = tuple(raw.dtype.names or ())
    if not {"Number", "LTP", "LTD"}.issubset(names):
        raise ValueError(f"{path.name}: columns Number, LTP and LTD are required; got {names}")
    number = np.atleast_1d(raw["Number"])
    ltp = np.atleast_1d(raw["LTP"]).astype(np.float64)
    ltd = np.atleast_1d(raw["LTD"]).astype(np.float64)
    if len(number) != 80 or not np.array_equal(number, np.arange(1, 81)):
        raise ValueError("Measured CSV must contain exactly Number=1..80; row 1 is zero-pulse state")
    if not np.all(np.isfinite(ltp)) or not np.all(np.isfinite(ltd)):
        raise ValueError("Measured conductance table contains missing/nonfinite states")
    if np.any(ltp <= 0) or np.any(ltd <= 0) or ltp[-1] <= ltp[0] or ltd[-1] >= ltd[0]:
        raise ValueError("Invalid positive LTP low-to-high or LTD high-to-low endpoint ordering")
    return ltp, ltd

class DeviceCurve:
    """Measured single-device states and exact differential-pair quantizer.

    Both measured branches are kept.  One physical conductance state can therefore be selected
    from either an LTP state or an LTD state.  All possible G+ - G- combinations are enumerated
    once (2N x 2N; N=80 -> 25,600 pair states), sorted, and searched by nearest DeltaG.
    """
    LTP_BRANCH = 0
    LTD_BRANCH = 1

    def __init__(self, path: Path, device: torch.device):
        ltp, ltd = read_device_csv(path)
        self.n_states = int(len(ltp))
        self.device = device
        self.ltp = torch.tensor(ltp, dtype=torch.float32, device=device)
        self.ltd = torch.tensor(ltd, dtype=torch.float32, device=device)  # high -> low

        self.g_min = float(min(ltp.min(), ltd.min()))
        self.g_max = float(max(ltp.max(), ltd.max()))
        self.g_span = float(self.g_max - self.g_min)
        if self.g_span <= 0:
            raise ValueError("Invalid measured conductance span")

        # A single physical device may be programmed to any state measured on either branch.
        single_g = np.concatenate([ltp, ltd]).astype(np.float32)
        single_branch = np.concatenate([
            np.zeros(self.n_states, dtype=np.int64),
            np.ones(self.n_states, dtype=np.int64),
        ])
        single_index = np.concatenate([
            np.arange(self.n_states, dtype=np.int64),
            np.arange(self.n_states, dtype=np.int64),
        ])

        # Enumerate every measured differential pair.  Keeping the branch/index metadata means
        # the selected signed weight can be converted back into an actual G+ and G- state.
        diff = (single_g[:, None] - single_g[None, :]).reshape(-1)
        p_state = np.repeat(np.arange(single_g.size, dtype=np.int64), single_g.size)
        n_state = np.tile(np.arange(single_g.size, dtype=np.int64), single_g.size)
        order = np.argsort(diff, kind="mergesort")

        self.single_g = torch.tensor(single_g, dtype=torch.float32, device=device)
        self.single_branch = torch.tensor(single_branch, dtype=torch.long, device=device)
        self.single_index = torch.tensor(single_index, dtype=torch.long, device=device)
        self.diff_values = torch.tensor(diff[order], dtype=torch.float32, device=device)
        self.diff_p_state = torch.tensor(p_state[order], dtype=torch.long, device=device)
        self.diff_n_state = torch.tensor(n_state[order], dtype=torch.long, device=device)
        self.delta_min = float(diff.min())
        self.delta_max = float(diff.max())
        self.delta_span = max(abs(self.delta_min), abs(self.delta_max))
        # The ideal LTP and LTD branches are the same 80 uniformly spaced states
        # traversed in opposite directions. Two such devices have 2*N-1 unique
        # differential conductances, not 2*(2*N)-1.
        self.ideal_levels = 2 * self.n_states - 1
        self.ideal_step = self.g_span / (self.n_states - 1)

        print(
            f"[device] {path.name}: {self.n_states} LTP + {self.n_states} LTD measured states; "
            f"G=[{self.g_min:.6e},{self.g_max:.6e}] S; dynamic range={self.g_max/self.g_min:.2f}x"
        )
        print(
            f"[device] differential trainable range DeltaG=[{self.delta_min:.6e},{self.delta_max:.6e}] S; "
            f"enumerated pair states={self.diff_values.numel():,}"
        )

    def nearest_pair(self, target_dg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return measured-state ids for G+, G- and the nearest achievable DeltaG."""
        x = target_dg.clamp(self.delta_min, self.delta_max).contiguous()
        pos = torch.searchsorted(self.diff_values, x)
        hi = pos.clamp(0, self.diff_values.numel() - 1)
        lo = (pos - 1).clamp(0, self.diff_values.numel() - 1)
        vlo = self.diff_values[lo]
        vhi = self.diff_values[hi]
        choose_hi = (vhi - x).abs() < (x - vlo).abs()
        k = torch.where(choose_hi, hi, lo)
        return self.diff_p_state[k], self.diff_n_state[k], self.diff_values[k]

    def state_conductance(self, state_id: torch.Tensor) -> torch.Tensor:
        return self.single_g[state_id]

    def state_programming_pulses(self, state_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical pulse count to reach a measured state from a known endpoint.

        LTP state i: reset to low endpoint with LTD, then apply i LTP pulses.
        LTD state i: set to high endpoint with LTP, then apply i LTD pulses.
        This makes the selected measured branch/state explicit and avoids the old path ambiguity.
        """
        branch = self.single_branch[state_id]
        idx = self.single_index[state_id]
        full = torch.full_like(idx, self.n_states - 1)
        ltp_p = torch.where(branch == self.LTP_BRANCH, idx, full)
        ltd_p = torch.where(branch == self.LTP_BRANCH, full, idx)
        return ltp_p, ltd_p


class DifferentialMemristorState(nn.Module):
    """A signed synapse represented directly as physical differential conductance DeltaG."""
    def __init__(self, shape: tuple[int, ...], curve: DeviceCurve, init_bound: float = 0.22):
        super().__init__()
        self.curve = curve
        init = torch.empty(shape, device=curve.device).uniform_(-init_bound, init_bound)
        # IMPORTANT: the trainable parameter itself is in siemens and is physically bounded.
        self.shadow_dg = nn.Parameter(init * curve.delta_span)

        zero_state = torch.zeros(shape, dtype=torch.long, device=curve.device)
        self.register_buffer("state_p", zero_state.clone())
        self.register_buffer("state_n", zero_state.clone())
        self.register_buffer("g_p", torch.full(shape, curve.g_min, device=curve.device))
        self.register_buffer("g_n", torch.full(shape, curve.g_min, device=curve.device))
        self.register_buffer("pulse_ltp", torch.zeros((), dtype=torch.long, device=curve.device))
        self.register_buffer("pulse_ltd", torch.zeros((), dtype=torch.long, device=curve.device))
        self.mode = "continuous"  # continuous | qat | programmed

    def continuous_dg(self) -> torch.Tensor:
        return self.shadow_dg.clamp(self.curve.delta_min, self.curve.delta_max)

    def quantized_dg(self) -> torch.Tensor:
        _, _, q = self.curve.nearest_pair(self.continuous_dg())
        return q

    def ideal_quantized_dg(self) -> torch.Tensor:
        dg = self.continuous_dg()
        # Explicit ideal differential pair: G+ and G- are selected from the
        # same reversible, uniformly spaced 0..N-1 conductance states.
        index = torch.round(dg / self.curve.ideal_step).clamp(
            -(self.curve.n_states - 1), self.curve.n_states - 1
        )
        g_p = self.curve.g_min + index.clamp_min(0) * self.curve.ideal_step
        g_n = self.curve.g_min + (-index).clamp_min(0) * self.curve.ideal_step
        return g_p - g_n

    def programmed_dg(self) -> torch.Tensor:
        return self.g_p - self.g_n

    def effective_z(self) -> torch.Tensor:
        dg = self.continuous_dg()
        if self.mode == "continuous":
            eff = dg
        elif self.mode == "ideal_qat":
            q = self.ideal_quantized_dg()
            eff = dg + (q - dg).detach()
        elif self.mode == "ideal":
            eff = self.ideal_quantized_dg()
        elif self.mode == "qat":
            q = self.quantized_dg()
            # measured-state forward, continuous-gradient backward
            eff = dg + (q - dg).detach()
        elif self.mode == "programmed":
            eff = self.programmed_dg()
        else:
            raise RuntimeError(f"Unknown memristor weight mode: {self.mode}")
        return (eff / self.curve.delta_span).clamp(-1.0, 1.0)

    @torch.no_grad()
    def project_shadow(self) -> None:
        self.shadow_dg.clamp_(self.curve.delta_min, self.curve.delta_max)

    @torch.no_grad()
    def set_mode(self, mode: str) -> None:
        if mode not in {"continuous", "ideal_qat", "ideal", "qat", "programmed"}:
            raise ValueError("unsupported weight mode")
        self.mode = mode

    @torch.no_grad()
    def program_from_shadow(self) -> None:
        """Map the trained DeltaG to exact measured discrete G+/G- states."""
        self.project_shadow()
        p_state, n_state, _ = self.curve.nearest_pair(self.shadow_dg)
        p_ltp, p_ltd = self.curve.state_programming_pulses(p_state)
        n_ltp, n_ltd = self.curve.state_programming_pulses(n_state)
        self.pulse_ltp.add_(p_ltp.sum(dtype=torch.long) + n_ltp.sum(dtype=torch.long))
        self.pulse_ltd.add_(p_ltd.sum(dtype=torch.long) + n_ltd.sum(dtype=torch.long))
        self.state_p.copy_(p_state)
        self.state_n.copy_(n_state)
        self.g_p.copy_(self.curve.state_conductance(p_state))
        self.g_n.copy_(self.curve.state_conductance(n_state))

    @torch.no_grad()
    def mapping_error(self, quantizer: str = "measured") -> tuple[float, float]:
        if quantizer not in {"ideal", "measured"}:
            raise ValueError(f"unsupported quantizer: {quantizer}")
        target = self.continuous_dg()
        q = self.ideal_quantized_dg() if quantizer == "ideal" else self.quantized_dg()
        err = (q - target).abs() / self.curve.delta_span
        return float(err.mean()), float(err.max())


class MemConv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, curve: DeviceCurve, padding: int = 1):
        super().__init__()
        fan_in = cin * k * k
        self.scale = math.sqrt(6.0 / fan_in)
        self.state = DifferentialMemristorState((cout, cin, k, k), curve)
        self.bias = nn.Parameter(torch.zeros(cout))
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.state.effective_z() * self.scale, self.bias, padding=self.padding)


class MemLinear(nn.Module):
    def __init__(self, nin: int, nout: int, curve: DeviceCurve):
        super().__init__()
        fan_in = nin
        self.scale = math.sqrt(6.0 / fan_in)
        self.state = DifferentialMemristorState((nout, nin), curve)
        self.bias = nn.Parameter(torch.zeros(nout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.state.effective_z() * self.scale, self.bias)

# =============================================================================
# Standard LIF neuron: beta=exp(-dt/tau)
# =============================================================================
class _FastSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, slope: float):
        ctx.save_for_backward(x)
        ctx.slope = float(slope)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        k = ctx.slope
        return grad_output / (1.0 + k * x.abs()).pow(2), None

def spike_fn(x: torch.Tensor, slope: float = 5.0) -> torch.Tensor:
    return _FastSpike.apply(x, slope)

class DeviceLIF(nn.Module):
    def __init__(self, beta: float, threshold: float, slope: float = 5.0):
        super().__init__()
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must lie in (0,1)")
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.slope = float(slope)

    def sequence(self, current_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem = torch.zeros_like(current_t[0])
        spikes = []
        for t in range(current_t.size(0)):
            mem = self.beta * mem + current_t[t]
            spk = spike_fn(mem - self.threshold, self.slope)
            mem = mem - spk.detach() * self.threshold
            spikes.append(spk)
        return torch.stack(spikes, 0), mem

# =============================================================================
# OCT data / preprocessing
# =============================================================================
def enhance_oct(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(0.08)
    z = ((x - mean) / (2.45 * std) + 0.5).clamp(0, 1)
    smooth = F.avg_pool2d(z, 3, 1, 1)
    return (z + 0.28 * (z - smooth)).clamp(0, 1)

def augment_oct(x: torch.Tensor) -> torch.Tensor:
    b = x.size(0)
    flip = torch.rand(b, device=x.device) < 0.5
    if torch.any(flip):
        x = x.clone()
        x[flip] = torch.flip(x[flip], dims=[3])

    if torch.rand((), device=x.device) < 0.60:
        dy = int(torch.randint(-1, 2, ()).item())
        dx = int(torch.randint(-1, 2, ()).item())
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        if dy > 0:
            x[:, :, :dy, :] = 0
        elif dy < 0:
            x[:, :, dy:, :] = 0
        if dx > 0:
            x[:, :, :, :dx] = 0
        elif dx < 0:
            x[:, :, :, dx:] = 0

    contrast = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.92, 1.08)
    brightness = torch.empty(b, 1, 1, 1, device=x.device).uniform_(-0.02, 0.02)
    x = ((x - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)

    if torch.rand((), device=x.device) < 0.25:
        x = (x + 0.006 * torch.randn_like(x)).clamp(0, 1)
    return x

class OCTDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = np.ascontiguousarray(images, dtype=np.uint8)
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.images[i].copy()).float().unsqueeze(0) / 255.0
        return x, int(self.labels[i])

def validation_keep_indices(train_images: np.ndarray, val_images: np.ndarray) -> np.ndarray:
    """Exclude only train-overlapping validation rows; retain internal duplicates."""
    train_pixels = {image.tobytes() for image in train_images}
    return np.fromiter(
        (i for i, image in enumerate(val_images) if image.tobytes() not in train_pixels),
        dtype=np.int64,
    )

def load_full_splits(cfg):
    npz = Path(cfg.dataset_path).expanduser() if cfg.dataset_path else SCRIPT_DIR / "data" / "octmnist.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"OCTMNIST archive not found: {npz}")
    expected = {"train": 97477, "val": 10832, "test": 1000}
    arrays = {}
    with np.load(npz, allow_pickle=False) as f:
        for split, count in expected.items():
            images = f[f"{split}_images"]
            labels = f[f"{split}_labels"].reshape(-1)
            if images.shape != (count, 28, 28) or labels.shape != (count,):
                raise ValueError(f"Unexpected {split} split shapes: {images.shape}, {labels.shape}")
            if not np.isin(labels, np.arange(N_CLASSES)).all():
                raise ValueError(f"Unexpected labels in {split} split")
            arrays[split] = (images, labels)

    train_images, train_labels = arrays["train"]
    val_images, val_labels = arrays["val"]
    test_images, test_labels = arrays["test"]
    keep = validation_keep_indices(train_images, val_images)
    exclusions_file = SCRIPT_DIR / "data" / "validation_excluded_indices.csv"
    if exclusions_file.is_file():
        with exclusions_file.open(newline="", encoding="utf-8-sig") as stream:
            listed = np.asarray([int(row["official_val_index"])
                                 for row in csv.DictReader(stream)], dtype=np.int64)
        computed = np.setdiff1d(np.arange(len(val_images)), keep, assume_unique=True)
        if not np.array_equal(listed, computed):
            raise ValueError("Published validation exclusion index file differs from the active filter")
    print(f"[data] validation: excluded {len(val_images) - len(keep)} exact train-image matches")
    for split, labels in (("train", train_labels), ("val", val_labels[keep]),
                          ("test", test_labels)):
        counts = np.bincount(labels.astype(np.int64), minlength=N_CLASSES)
        print(f"[data] {split}={len(labels)}; classes={counts.tolist()}")
    return (OCTDataset(train_images, train_labels),
            OCTDataset(val_images[keep], val_labels[keep]),
            OCTDataset(test_images, test_labels))

class FullSplitBatchSampler(Sampler):
    def __init__(self, size: int, batch_size: int, rank: int, world_size: int, seed: int):
        if size < world_size or batch_size < world_size:
            raise ValueError("Dataset and global batch size must be at least WORLD_SIZE")
        self.size = size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        batches = [indices[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) < self.world_size:
            batches[-2].extend(batches.pop())
        for batch in batches:
            yield batch[self.rank::self.world_size]

    def __len__(self) -> int:
        count = (self.size + self.batch_size - 1) // self.batch_size
        remainder = self.size % self.batch_size
        if count > 1 and 0 < remainder < self.world_size:
            count -= 1
        return count

def make_loaders(train, val, test, cfg, rank: int = 0, world_size: int = 1):
    kw = dict(num_workers=cfg.workers, pin_memory=torch.cuda.is_available())
    if world_size > 1:
        sampler = FullSplitBatchSampler(len(train), cfg.batch_size, rank, world_size, cfg.seed)
        train_loader = DataLoader(train, batch_sampler=sampler, **kw)
    else:
        train_loader = DataLoader(
            train, batch_size=cfg.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(cfg.seed), **kw
        )
    val_loader = DataLoader(val, batch_size=cfg.eval_batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test, batch_size=cfg.eval_batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader

# =============================================================================
# SNN (Compact VGG-style structure with ascending thresholds)
# =============================================================================
class OCTMemristorSNN(nn.Module):
    def __init__(self, curve: DeviceCurve, cfg):
        super().__init__()
        self.curve = curve
        self.dt_s = float(cfg.dt_s)
        self.beta = beta_from_dt(cfg.dt_s)
        self.input_rate_hz = float(cfg.input_rate_hz)

        # Stronger, still compact enough for 28x28 OCTMNIST.
        # Two spike-encoded input channels = raw OCT + contrast-enhanced OCT.
        self.c11 = MemConv(2, 48, 3, curve, 1)
        self.b11 = nn.BatchNorm2d(48)
        self.l11 = DeviceLIF(self.beta, 0.90)
        self.c12 = MemConv(48, 48, 3, curve, 1)
        self.b12 = nn.BatchNorm2d(48)
        self.l12 = DeviceLIF(self.beta, 0.95)

        self.c21 = MemConv(48, 96, 3, curve, 1)
        self.b21 = nn.BatchNorm2d(96)
        self.l21 = DeviceLIF(self.beta, 1.00)
        self.c22 = MemConv(96, 96, 3, curve, 1)
        self.b22 = nn.BatchNorm2d(96)
        self.l22 = DeviceLIF(self.beta, 1.05)

        self.c31 = MemConv(96, 192, 3, curve, 1)
        self.b31 = nn.BatchNorm2d(192)
        self.l31 = DeviceLIF(self.beta, 1.10)
        self.c32 = MemConv(192, 192, 3, curve, 1)
        self.b32 = nn.BatchNorm2d(192)
        self.l32 = DeviceLIF(self.beta, 1.15)
        self.c33 = MemConv(192, 192, 3, curve, 1)
        self.b33 = nn.BatchNorm2d(192)
        self.l33 = DeviceLIF(self.beta, 1.20)

        self.fc1 = MemLinear(192 * 3 * 3, 384, curve)
        self.ln1 = nn.LayerNorm(384)
        self.lf1 = DeviceLIF(self.beta, 1.05)
        self.out = MemLinear(384, N_CLASSES, curve)

    @staticmethod
    def _apply_time(module: nn.Module, x_t: torch.Tensor) -> torch.Tensor:
        t, b = x_t.shape[:2]
        y = module(x_t.flatten(0, 1))
        return y.view(t, b, *y.shape[1:])

    def phase_spike_encode(self, image: torch.Tensor, steps: int) -> torch.Tensor:
        p = (1.0 - torch.exp(-self.input_rate_hz * self.dt_s * image)).clamp(0.0, 0.97)
        phase = torch.full_like(p, 0.5)
        seq = []
        for _ in range(steps):
            phase = phase + p
            spk = (phase >= 1.0).to(image.dtype)
            phase = phase - spk
            seq.append(spk)
        return torch.stack(seq, 0)

    def forward(self, image: torch.Tensor, steps: int, return_rates: bool = False,
                return_last_spikes: bool = False):
        if steps != 16:
            raise ValueError(f"This paper configuration uses exactly 16 SNN steps; got {steps}")

        raw = image.clamp(0, 1)
        enhanced = enhance_oct(raw)
        encoded_image = torch.cat([raw, enhanced], dim=1)
        x = self.phase_spike_encode(encoded_image, steps)

        # Block 1: 28 -> 14. Equal-width second conv receives a residual spike current.
        i = self._apply_time(self.b11, self._apply_time(self.c11, x))
        s11, _ = self.l11.sequence(i)
        i = self._apply_time(self.b12, self._apply_time(self.c12, s11)) + 0.20 * s11
        s12, _ = self.l12.sequence(i)
        p1 = F.avg_pool2d(s12.flatten(0, 1), 2, 2)
        p1 = p1.view(steps, image.size(0), 48, 14, 14) * 4.0

        # Block 2: 14 -> 7
        i = self._apply_time(self.b21, self._apply_time(self.c21, p1))
        s21, _ = self.l21.sequence(i)
        i = self._apply_time(self.b22, self._apply_time(self.c22, s21)) + 0.20 * s21
        s22, _ = self.l22.sequence(i)
        p2 = F.avg_pool2d(s22.flatten(0, 1), 2, 2)
        p2 = p2.view(steps, image.size(0), 96, 7, 7) * 4.0

        # Block 3: 7 -> 3
        i = self._apply_time(self.b31, self._apply_time(self.c31, p2))
        s31, _ = self.l31.sequence(i)
        i = self._apply_time(self.b32, self._apply_time(self.c32, s31)) + 0.15 * s31
        s32, _ = self.l32.sequence(i)
        i = self._apply_time(self.b33, self._apply_time(self.c33, s32)) + 0.15 * s32
        s33, _ = self.l33.sequence(i)
        p3 = F.avg_pool2d(s33.flatten(0, 1), 2, 2)
        p3 = p3.view(steps, image.size(0), 192, 3, 3) * 4.0

        pooled = p3.view(steps, image.size(0), 192 * 3 * 3)
        i = self._apply_time(self.ln1, self._apply_time(self.fc1, pooled))
        sf, _ = self.lf1.sequence(i)
        out_i = self._apply_time(self.out, sf)

        # Leaky temporal evidence; a monotonic tanh bound leaves argmax unchanged.
        mem = torch.zeros_like(out_i[0])
        evidence = torch.zeros_like(mem)
        for t in range(steps):
            mem = self.beta * mem + out_i[t]
            evidence = evidence + mem
        raw_logits = evidence / float(steps)
        logits = 4.0 * torch.tanh(raw_logits / 4.0)

        if return_rates:
            rates = torch.stack([
                s11.mean(), s12.mean(), s21.mean(), s22.mean(),
                s31.mean(), s32.mean(), s33.mean(), sf.mean()
            ])
            if return_last_spikes:
                return logits, rates, s33
            return logits, rates
        if return_last_spikes:
            return logits, s33
        return logits

    def _mem_states(self):
        for m in self.modules():
            if isinstance(m, (MemConv, MemLinear)):
                yield m.state

    @torch.no_grad()
    def set_weight_mode(self, mode: str) -> None:
        for st in self._mem_states():
            st.set_mode(mode)

    @torch.no_grad()
    def project_shadow_weights(self) -> None:
        for st in self._mem_states():
            st.project_shadow()

    @torch.no_grad()
    def program_all_from_shadow(self) -> None:
        for st in self._mem_states():
            st.program_from_shadow()

    @torch.no_grad()
    def quantization_stats(self, quantizer: str = "measured") -> tuple[float, float]:
        means, maxes = [], []
        for st in self._mem_states():
            a, b = st.mapping_error(quantizer)
            means.append(a)
            maxes.append(b)
        return float(np.mean(means)), float(np.max(maxes))

    def shadow_dg_parameters(self) -> list[nn.Parameter]:
        return [st.shadow_dg for st in self._mem_states()]

    def non_shadow_parameters(self) -> list[nn.Parameter]:
        dg_ids = {id(p) for p in self.shadow_dg_parameters()}
        return [p for p in self.parameters() if id(p) not in dg_ids]

    def pulse_counts(self) -> tuple[int, int]:
        p = d = 0
        for st in self._mem_states():
            p += int(st.pulse_ltp.item())
            d += int(st.pulse_ltd.item())
        return p, d

# =============================================================================
# Configuration / training
# =============================================================================
@dataclass
class Config:
    seed: int = 19
    dataset_path: Optional[str] = None
    dt_s: float = 1e-6
    steps: int = 16
    input_rate_hz: float = 9.0e5
    warmup_epochs: int = 70
    device_epochs: int = 30
    extra_device_epochs: int = 0
    batch_size: int = 96
    eval_batch_size: int = 384
    workers: int = 0
    warmup_lr: float = 8.0e-4          # normalized/base LR for AdamW
    device_lr: float = 2.0e-4          # QAT base LR
    weight_decay: float = 1.5e-4
    grad_clip: float = 3.0
    label_smoothing: float = 0.035
    class_weights: tuple[float, float, float, float] = (0.69666175, 1.26143235, 1.44769719, 0.59420871)
    spike_reg: float = 2.5e-4
    firing_low: float = 0.035
    firing_high: float = 0.36
    target_acc: float = 0.90
    target_min_recall: float = 0.80
    device_patience: int = 12
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_optimizer(model: OCTMemristorSNN, base_lr: float, weight_decay: float):
    """AdamW with a physical-unit LR for DeltaG and a normal LR for digital scalars.

    Adam's update magnitude is approximately set by lr, so DeltaG (stored in siemens) receives
    base_lr * DeltaG_range, while BN/LN/bias parameters receive base_lr.
    """
    dg = model.shadow_dg_parameters()
    other = model.non_shadow_parameters()
    return torch.optim.AdamW([
        {"params": dg, "lr": base_lr * model.curve.delta_span, "weight_decay": 0.0, "name": "DeltaG"},
        {"params": other, "lr": base_lr, "weight_decay": weight_decay, "name": "other"},
    ])


def classification_loss(logits: torch.Tensor, y: torch.Tensor,
                        class_weights: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Weighted cross-entropy with manual label smoothing.

    This implementation deliberately avoids the ``label_smoothing=`` keyword of
    ``torch.nn.functional.cross_entropy`` so the script also runs on older PyTorch
    releases (the user's Anaconda environment raises TypeError for that keyword).

    The smoothed target is
        q = (1-eps) * one_hot(y) + eps / C,
    and class weighting is applied to the whole target distribution.  For eps=0
    this reduces to ordinary weighted cross-entropy.
    """
    eps = float(max(0.0, min(1.0, smoothing)))
    if eps <= 0.0:
        return F.cross_entropy(logits, y, weight=class_weights)

    logp = F.log_softmax(logits, dim=1)
    ncls = int(logits.size(1))
    with torch.no_grad():
        q = torch.full_like(logp, eps / float(ncls))
        q.scatter_(1, y.view(-1, 1), 1.0 - eps + eps / float(ncls))

    # Weight each class contribution.  Normalize by the effective target weight
    # so the loss scale stays comparable to ordinary weighted CE.
    weighted_q = q * class_weights.view(1, -1)
    per_sample = -(weighted_q * logp).sum(dim=1)
    normalizer = weighted_q.sum(dim=1).clamp_min(1e-12)
    return (per_sample / normalizer).mean()

def confusion_metrics(pred: torch.Tensor, true: torch.Tensor):
    cm = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long)
    for t, p in zip(true.cpu(), pred.cpu()):
        cm[int(t), int(p)] += 1
    tp = cm.diag().float()
    recall = tp / cm.sum(1).clamp_min(1)
    precision = tp / cm.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
    acc = float(tp.sum() / cm.sum().clamp_min(1))
    return dict(
        acc=acc, cm=cm, precision=precision, recall=recall, f1=f1,
        macro_f1=float(f1.mean()), min_recall=float(recall.min())
    )

@torch.no_grad()
def collect_logits(model, loader, device, steps: int, tta_flip: bool = False):
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        z = model(x, steps)
        if tta_flip:
            z = 0.5 * (z + model(torch.flip(x, dims=[3]), steps))
        zs.append(z.cpu())
        ys.append(y.cpu())
    return torch.cat(zs), torch.cat(ys)

def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, calibration=None):
    z = logits
    if calibration is not None:
        temperature, bias = calibration
        z = z / float(temperature) + bias[None]
    pred = z.argmax(1)
    m = confusion_metrics(pred, y)
    m["loss"] = float(F.cross_entropy(z, y))
    return m

@torch.no_grad()
def evaluate(model, loader, device, steps: int, tta_flip: bool = False, calibration=None):
    z, y = collect_logits(model, loader, device, steps, tta_flip)
    return metrics_from_logits(z, y, calibration)

def selection_score(m: dict) -> float:
    return 0.66 * m["acc"] + 0.19 * m["macro_f1"] + 0.15 * m["min_recall"]

def print_metrics(tag: str, m: dict) -> None:
    print(
        f"[{tag}] loss={m['loss']:.4f} acc={100*m['acc']:.2f}% "
        f"macroF1={100*m['macro_f1']:.2f}% minRecall={100*m['min_recall']:.2f}%"
    )
    print(f"[{tag}] confusion matrix rows=true, cols=pred:\n{m['cm'].numpy()}")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"[{tag}] {name:>6s}: P={100*float(m['precision'][i]):6.2f}% "
            f"R={100*float(m['recall'][i]):6.2f}% F1={100*float(m['f1'][i]):6.2f}%"
        )


# =============================================================================
# CSV data export helpers (recording only; no training/inference logic changes)
# =============================================================================
def write_dict_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dictionaries to CSV with stable column order."""
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[csv] saved -> {path.name}")


def make_training_history_row(stage: str, weight_mode: str, stage_epoch: int,
                              global_epoch: int, train_loss: float, train_acc: float,
                              val: dict, score: float, lr: float, rates: np.ndarray,
                              pulse_budget: int = 0) -> dict:
    """Create one epoch record without affecting optimization or model state."""
    row = dict(
        stage=stage,
        weight_mode=weight_mode,
        stage_epoch=int(stage_epoch),
        global_epoch=int(global_epoch),
        train_loss=float(train_loss),
        train_acc=float(train_acc),
        train_acc_percent=100.0 * float(train_acc),
        val_loss=float(val["loss"]),
        val_acc=float(val["acc"]),
        val_acc_percent=100.0 * float(val["acc"]),
        val_macro_f1=float(val["macro_f1"]),
        val_macro_f1_percent=100.0 * float(val["macro_f1"]),
        val_min_recall=float(val["min_recall"]),
        val_min_recall_percent=100.0 * float(val["min_recall"]),
        selection_score=float(score),
        learning_rate=float(lr),
        device_pulse_budget=int(pulse_budget),
    )
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    for i, name in enumerate(LIF_LAYER_NAMES):
        row[f"rate_{name}"] = float(rates[i])
    return row


def save_training_and_lif_history(history: list[dict], output_dir: Path) -> None:
    """Export float/device epoch metrics and LIF firing-rate histories."""
    if not history:
        print("[csv] no training history in this run (for example --eval-only); history CSVs skipped")
        return

    metric_fields = [
        "stage", "weight_mode", "stage_epoch", "global_epoch",
        "train_loss", "train_acc", "train_acc_percent",
        "val_loss", "val_acc", "val_acc_percent",
        "val_macro_f1", "val_macro_f1_percent",
        "val_min_recall", "val_min_recall_percent",
        "selection_score", "learning_rate", "device_pulse_budget",
    ]
    rate_fields = [f"rate_{name}" for name in LIF_LAYER_NAMES]

    # Separate files make the ideal floating-point and mapped-device curves
    # directly usable for plotting, while the combined file preserves chronology.
    write_dict_rows_csv(
        output_dir / "paper_training_history_all.csv", history, metric_fields + rate_fields
    )
    float_rows = [r for r in history if r["weight_mode"] == "ideal_float_shadow"]
    device_rows = [r for r in history if r["weight_mode"] != "ideal_float_shadow"]
    if float_rows:
        write_dict_rows_csv(
            output_dir / "paper_float_weight_training_history.csv",
            float_rows, metric_fields + rate_fields
        )
    if device_rows:
        write_dict_rows_csv(
            output_dir / "paper_quantized_weight_training_history.csv",
            device_rows, metric_fields + rate_fields
        )

    # Long-format per-epoch LIF rates are convenient for Origin/Excel plotting.
    lif_epoch_rows = []
    for r in history:
        for name in LIF_LAYER_NAMES:
            lif_epoch_rows.append(dict(
                stage=r["stage"],
                weight_mode=r["weight_mode"],
                stage_epoch=r["stage_epoch"],
                global_epoch=r["global_epoch"],
                lif_layer=name,
                mean_firing_rate=float(r[f"rate_{name}"]),
                mean_firing_rate_percent=100.0 * float(r[f"rate_{name}"]),
            ))
    write_dict_rows_csv(
        output_dir / "paper_lif_firing_rates_by_epoch.csv", lif_epoch_rows,
        ["stage", "weight_mode", "stage_epoch", "global_epoch", "lif_layer",
         "mean_firing_rate", "mean_firing_rate_percent"]
    )

    # Full-training averages: mean of the recorded epoch-average firing rates.
    lif_full_rows = []
    for name in LIF_LAYER_NAMES:
        all_values = np.asarray([r[f"rate_{name}"] for r in history], dtype=np.float64)
        shadow_values = np.asarray(
            [r[f"rate_{name}"] for r in float_rows], dtype=np.float64
        )
        device_values = np.asarray(
            [r[f"rate_{name}"] for r in device_rows], dtype=np.float64
        )
        lif_full_rows.append(dict(
            lif_layer=name,
            full_training_mean_firing_rate=float(all_values.mean()),
            full_training_mean_firing_rate_percent=100.0 * float(all_values.mean()),
            float_stage_mean_firing_rate=(float(shadow_values.mean()) if shadow_values.size else ""),
            float_stage_mean_firing_rate_percent=(100.0 * float(shadow_values.mean()) if shadow_values.size else ""),
            device_stage_mean_firing_rate=(float(device_values.mean()) if device_values.size else ""),
            device_stage_mean_firing_rate_percent=(100.0 * float(device_values.mean()) if device_values.size else ""),
            recorded_epochs=int(all_values.size),
            float_stage_epochs=int(shadow_values.size),
            device_stage_epochs=int(device_values.size),
        ))
    write_dict_rows_csv(
        output_dir / "paper_lif_firing_rates_full_training.csv", lif_full_rows,
        ["lif_layer",
         "full_training_mean_firing_rate", "full_training_mean_firing_rate_percent",
         "float_stage_mean_firing_rate", "float_stage_mean_firing_rate_percent",
         "device_stage_mean_firing_rate", "device_stage_mean_firing_rate_percent",
         "recorded_epochs", "float_stage_epochs", "device_stage_epochs"]
    )


def save_confusion_matrix_csv(path: Path, metrics: dict) -> None:
    """Save one labelled true-class x predicted-class confusion matrix."""
    cm = metrics["cm"].detach().cpu().numpy().astype(np.int64)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["true_class"] + [f"pred_{name}" for name in CLASS_NAMES] + ["row_total"])
        for i, name in enumerate(CLASS_NAMES):
            w.writerow([name] + cm[i].tolist() + [int(cm[i].sum())])
        w.writerow(["column_total"] + cm.sum(axis=0).tolist() + [int(cm.sum())])
    print(f"[csv] saved -> {path.name}")


def save_final_split_metrics_csv(path: Path, split_metrics: list[tuple[str, dict]]) -> None:
    """Save final train/validation/test summary metrics beside confusion matrices."""
    rows = []
    for split, m in split_metrics:
        row = dict(
            split=split, loss=float(m["loss"]), acc=float(m["acc"]),
            acc_percent=100.0 * float(m["acc"]),
            macro_f1=float(m["macro_f1"]), macro_f1_percent=100.0 * float(m["macro_f1"]),
            min_recall=float(m["min_recall"]), min_recall_percent=100.0 * float(m["min_recall"]),
        )
        for i, name in enumerate(CLASS_NAMES):
            row[f"precision_{name}"] = float(m["precision"][i])
            row[f"recall_{name}"] = float(m["recall"][i])
            row[f"f1_{name}"] = float(m["f1"][i])
        rows.append(row)
    fields = ["split", "loss", "acc", "acc_percent", "macro_f1", "macro_f1_percent",
              "min_recall", "min_recall_percent"]
    for name in CLASS_NAMES:
        fields.extend([f"precision_{name}", f"recall_{name}", f"f1_{name}"])
    write_dict_rows_csv(path, rows, fields)

def train_epoch(model, forward_model, loader, optimizer, cfg, device, weight_mode: str,
                epoch: int = 0, rank: int = 0, world_size: int = 1):
    forward_model.train()
    model.set_weight_mode(weight_mode)
    if world_size > 1:
        loader.batch_sampler.set_epoch(epoch)
    weights = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    n = correct = 0
    loss_sum = 0.0
    rate_sum = torch.zeros(8, device=device)
    t0 = time.time()

    for bi, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        x = augment_oct(x)
        optimizer.zero_grad(set_to_none=True)
        logits, rates = forward_model(x, cfg.steps, return_rates=True)
        cls = classification_loss(logits, y, weights, cfg.label_smoothing)
        low_pen = F.relu(cfg.firing_low - rates).pow(2)
        high_pen = F.relu(rates - cfg.firing_high).pow(2)
        rate_loss = (low_pen + high_pen).mean()
        loss = cls + cfg.spike_reg * rate_loss

        if (not torch.isfinite(loss)) or float(loss.detach()) >= 10.0:
            raise RuntimeError(
                f"Training loss left the required finite <10 range: {float(loss.detach()):.6f}. "
                "Check data normalization / learning rate."
            )

        bs = y.numel()
        if world_size > 1:
            global_bs = (cfg.batch_size if bi < len(loader) else
                         len(loader.dataset) - cfg.batch_size * (len(loader) - 1))
            backward_loss = loss * (world_size * bs / global_bs)
        else:
            backward_loss = loss
        backward_loss.backward()
        # DeltaG gradients are expressed per siemens and are naturally much larger numerically.
        torch.nn.utils.clip_grad_norm_(model.shadow_dg_parameters(), cfg.grad_clip / model.curve.delta_span)
        torch.nn.utils.clip_grad_norm_(model.non_shadow_parameters(), cfg.grad_clip)
        optimizer.step()
        model.project_shadow_weights()

        n += bs
        correct += int((logits.argmax(1) == y).sum().item())
        loss_sum += float(loss.item()) * bs
        rate_sum += rates.detach() * bs

        if rank == 0 and bi % 100 == 0:
            dt = max(time.time() - t0, 1e-6)
            mode = weight_mode
            print(
                f"  batch {bi:03d}/{len(loader)} mode={mode} loss={loss_sum/n:.4f} "
                f"acc={100*correct/n:.2f}% {n/dt:.0f} img/s",
                flush=True,
            )

    if world_size > 1:
        stats = torch.cat((torch.tensor([loss_sum, correct, n], device=device, dtype=torch.float64),
                           rate_sum.double()))
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, correct, n = stats[:3].tolist()
        rate_sum = stats[3:]
    return loss_sum / n, correct / n, (rate_sum / n).cpu().numpy()

def save_ckpt(path: Path, model, cfg, epoch_tag: str, val: dict):
    torch.save(
        dict(model=model.state_dict(), config=asdict(cfg), epoch_tag=epoch_tag,
             val_acc=val["acc"], val_macro_f1=val["macro_f1"],
             val_min_recall=val["min_recall"]),
        path,
    )

def fit_validation_calibration(logits: torch.Tensor, y: torch.Tensor):
    """Validation-only temperature + 4 logit biases; no test labels are used."""
    z = logits.detach().float()
    y = y.long()
    log_temp = torch.zeros((), requires_grad=True)
    raw_bias = torch.zeros(N_CLASSES, requires_grad=True)
    opt = torch.optim.LBFGS([log_temp, raw_bias], lr=0.12, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        temp = log_temp.exp().clamp(0.35, 3.0)
        bias = raw_bias - raw_bias.mean()
        out = z / temp + bias[None]
        loss = F.cross_entropy(out, y) + 0.015 * bias.pow(2).mean()
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except RuntimeError as e:
        print(f"[calibration] LBFGS warning: {e}; using identity calibration")
        return 1.0, torch.zeros(N_CLASSES)

    temp = float(log_temp.detach().exp().clamp(0.35, 3.0))
    bias = (raw_bias.detach() - raw_bias.detach().mean()).cpu()
    print(f"[calibration] T={temp:.4f}, class-bias={np.round(bias.numpy(),4)}")
    return temp, bias

# =============================================================================
# SNN Grad-CAM receptive fields
# =============================================================================
def snn_gradcam(model, x: torch.Tensor, cls: int, steps: int):
    model.eval()
    x = x.detach().requires_grad_(True)
    logits, s3 = model(x, steps, return_last_spikes=True)
    score = logits[0, cls]
    grad = torch.autograd.grad(score, s3, retain_graph=False, create_graph=False)[0]
    act = s3.mean(0)[0]
    weights = grad.mean(dim=(0, 3, 4))[0]
    cam = F.relu((weights[:, None, None] * act).sum(0))[None, None]
    cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.detach().cpu().numpy()

def plot_receptive_fields(model, ds: OCTDataset, device, cfg, calibration, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] RF skipped: {e}")
        return

    model.eval()
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    best = [(-1e9, None) for _ in range(N_CLASSES)]
    temp, bias = calibration

    with torch.no_grad():
        for x, y in loader:
            z = model(x.to(device), cfg.steps).cpu()
            z = z / temp + bias[None]
            prob = F.softmax(z, 1)
            pred = prob.argmax(1)
            for j in range(y.numel()):
                c = int(y[j])
                if int(pred[j]) == c and float(prob[j, c]) > best[c][0]:
                    best[c] = (float(prob[j, c]), x[j:j+1].clone())

    cams, imgs, confs = [], [], []
    for c in range(N_CLASSES):
        conf, x = best[c]
        if x is None:
            idx = int(np.flatnonzero(ds.labels == c)[0])
            x = ds[idx][0][None]
            conf = float("nan")
        cams.append(snn_gradcam(model, x.to(device), c, cfg.steps))
        imgs.append(x[0, 0].numpy())
        confs.append(conf)

    stack = np.stack(cams)
    global_max = max(float(np.percentile(stack, 99.5)), 1e-8)
    stack = np.clip(stack / global_max, 0, 1)

    fig, axes = plt.subplots(3, N_CLASSES, figsize=(12.5, 8.0))
    for c, name in enumerate(CLASS_NAMES):
        axes[0, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(f"{name}\ncorrect confidence={confs[c]:.3f}")
        axes[1, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
        axes[1, c].imshow(stack[c], cmap="jet", vmin=0, vmax=1, alpha=0.48)
        axes[2, c].imshow(stack[c], cmap="inferno", vmin=0, vmax=1)
        for r in range(3):
            axes[r, c].axis("off")

    axes[0, 0].set_ylabel("Representative OCT")
    axes[1, 0].set_ylabel("SNN Grad-CAM overlay")
    axes[2, 0].set_ylabel("Class RF / CAM")
    fig.suptitle("Class-discriminative SNN receptive fields (validation representatives)")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] receptive fields -> {path.name}")

def plot_device(curve: DeviceCurve, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    ltp = curve.ltp.detach().cpu().numpy()
    ltd = curve.ltd.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(np.arange(1, len(ltp)+1), ltp, marker="o", ms=2, lw=1, label="measured LTP")
    ax.plot(np.arange(1, len(ltd)+1), ltd, marker="o", ms=2, lw=1, label="measured LTD")
    ax.set(xlabel="Sequential programming pulse index", ylabel="Conductance (S)")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

# =============================================================================
# dt pilot sweep: validation only, never test
# =============================================================================
def dt_pilot_sweep(base_cfg: Config, candidates_us=(0.5, 1.0, 2.0, 4.0), pilot_epochs=2):
    """Optional validation-only dt sweep; the SNN length remains exactly 16 steps."""
    print("[dt-sweep] validation-only pilot; every candidate still uses exactly 16 SNN steps")
    rows = []
    for us in candidates_us:
        dt = us * 1e-6
        cfg = replace(
            base_cfg, dt_s=dt, steps=16,
            warmup_epochs=pilot_epochs, device_epochs=0,
            batch_size=min(128, base_cfg.batch_size),
        )
        seed_everything(cfg.seed)
        device = torch.device(cfg.device)
        curve = DeviceCurve(find_input_file("line_LTP_LTD.csv", ["line_LTP_LTD*.csv"]), device)
        tr, va, te = load_full_splits(cfg)
        tl, vl, _ = make_loaders(tr, va, te, cfg)
        model = OCTMemristorSNN(curve, cfg).to(device)
        model.set_weight_mode("continuous")
        opt = build_optimizer(model, cfg.warmup_lr, cfg.weight_decay)
        for ep in range(pilot_epochs):
            train_epoch(model, model, tl, opt, cfg, device, weight_mode="continuous", epoch=ep)
        val = evaluate(model, vl, device, cfg.steps)
        rows.append((us, beta_from_dt(dt), cfg.steps, val["acc"], val["min_recall"]))
        print(
            f"[dt-sweep] dt={us:.1f}us beta={beta_from_dt(dt):.6f} steps=16 "
            f"val={100*val['acc']:.2f}% minR={100*val['min_recall']:.2f}%"
        )
        del model
    return rows


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Shared 70-epoch OCTMNIST SNN baseline and 30-epoch ablation continuations")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--experiment", choices=("baseline", "float", "ideal", "measured"), default="baseline")
    p.add_argument("--base-checkpoint", type=str, default=None,
                   help="Required for a 30-epoch ablation branch; must be the epoch-70 LAST checkpoint")
    p.add_argument("--dt-us", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=70)
    p.add_argument("--device-epochs", type=int, default=30)
    p.add_argument("--extra-device-epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--local-rank", "--local_rank", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default="base_epoch_70.pt")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dt-sweep", action="store_true")
    p.add_argument("--pilot-epochs", type=int, default=2)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--early-stop", action="store_true", help="deprecated; rejected because all branches must run 30 epochs")
    p.add_argument("--report-tta", action="store_true", help="supplementary only; primary result is raw single-pass physical SNN")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def smoke_test(cfg: Config):
    device = torch.device(cfg.device)
    curve = DeviceCurve(find_input_file("line_LTP_LTD.csv", ["line_LTP_LTD*.csv"]), device)
    model = OCTMemristorSNN(curve, cfg).to(device)
    x = torch.rand(2, 1, 28, 28, device=device)
    y = torch.tensor([0, 1], device=device)

    model.set_weight_mode("continuous")
    z, r = model(x, steps=16, return_rates=True)
    loss = F.cross_entropy(z, y)
    loss.backward()
    model.project_shadow_weights()

    model.set_weight_mode("ideal_qat")
    with torch.no_grad():
        zi_qat = model(x, steps=16)
    ideal_mean_err, ideal_max_err = model.quantization_stats("ideal")
    model.set_weight_mode("ideal")
    with torch.no_grad():
        zi = model(x, steps=16)
    ideal_delta = float((zi_qat - zi).abs().max())

    model.set_weight_mode("qat")
    with torch.no_grad():
        zq = model(x, steps=16)
    mean_err, max_err = model.quantization_stats()

    model.program_all_from_shadow()
    model.set_weight_mode("programmed")
    with torch.no_grad():
        zp = model(x, steps=16)
    delta = float((zq - zp).abs().max())
    assert z.shape == (2, 4) and torch.isfinite(zp).all()
    assert ideal_delta < 2e-5, f"ideal QAT/inference mismatch: {ideal_delta}"
    assert delta < 2e-5, f"QAT/programmed mismatch: {delta}"
    assert float(loss.detach()) < 10.0
    print(
        f"[smoke] OK; steps=16 beta={model.beta:.9f}; CE={float(loss.detach()):.4f}; "
        f"ideal map error mean/max={ideal_mean_err:.3e}/{ideal_max_err:.3e}; "
        f"measured map error mean/max={mean_err:.3e}/{max_err:.3e}; "
        f"max |ideal-QAT/ideal logits|={ideal_delta:.3e}; "
        f"max |QAT-programmed logits|={delta:.3e}; rates={np.round(r.detach().cpu().numpy(),3)}"
    )


def report_final_outputs(model, curve, cfg, args, output_dir, checkpoint_path,
                         train_ds, val_ds, train_loader, val_loader, test_loader,
                         device, final_weight_mode):
    """Keep the existing final split, confusion, metadata and optional figure outputs."""
    model.set_weight_mode(final_weight_mode)
    val_logits, val_y = collect_logits(model, val_loader, device, cfg.steps, tta_flip=False)
    val_final = metrics_from_logits(val_logits, val_y, calibration=None)
    print_metrics(f"VAL-final {args.experiment} 16-step", val_final)

    test_logits, test_y = collect_logits(model, test_loader, device, cfg.steps, tta_flip=False)
    test_final = metrics_from_logits(test_logits, test_y, calibration=None)
    print_metrics(f"TEST {args.experiment} 16-step", test_final)

    train_eval_loader = DataLoader(
        train_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=torch.cuda.is_available()
    )
    train_logits, train_y = collect_logits(model, train_eval_loader, device, cfg.steps, tta_flip=False)
    train_final = metrics_from_logits(train_logits, train_y, calibration=None)
    print_metrics(f"TRAIN-final {args.experiment} 16-step", train_final)

    save_confusion_matrix_csv(output_dir / "paper_confusion_matrix_train.csv", train_final)
    save_confusion_matrix_csv(output_dir / "paper_confusion_matrix_validation.csv", val_final)
    save_confusion_matrix_csv(output_dir / "paper_confusion_matrix_test.csv", test_final)
    save_final_split_metrics_csv(
        output_dir / "paper_final_split_metrics.csv",
        [("train", train_final), ("validation", val_final), ("test", test_final)]
    )
    if args.report_tta:
        tta = evaluate(model, test_loader, device, cfg.steps, tta_flip=True)
        print_metrics("TEST supplementary flip-TTA", tta)
    if args.experiment == "measured":
        p_ltp, p_ltd = model.pulse_counts()
        print(f"[device] canonical programming pulse events: LTP={p_ltp:,}, LTD={p_ltd:,}")

    passed = (
        test_final["acc"] >= cfg.target_acc and
        test_final["min_recall"] >= cfg.target_min_recall and
        test_final["loss"] < 10.0
    )
    print(
        f"[target] {args.experiment} accuracy>=90%, every-class recall>=80%, loss<10: "
        f"{'PASS' if passed else 'NOT REACHED'}"
    )
    torch.save(dict(
        checkpoint=str(checkpoint_path.name), experiment=args.experiment, config=asdict(cfg),
        test_acc=test_final["acc"], test_loss=test_final["loss"],
        test_recall=test_final["recall"], test_cm=test_final["cm"], passed=passed,
        device_g_min=curve.g_min, device_g_max=curve.g_max,
        differential_delta_min=curve.delta_min, differential_delta_max=curve.delta_max,
        ideal_uniform_levels=curve.ideal_levels,
        measured_ordered_pair_states=int(curve.diff_values.numel()),
    ), output_dir / "paper_final_metadata.pt")
    if not args.no_plots:
        plot_device(curve, output_dir / "paper_device_curves.png")
        identity_calibration = (1.0, torch.zeros(N_CLASSES))
        plot_receptive_fields(model, val_ds, device, cfg, identity_calibration, output_dir / "new2.png")


def run_main(args, rank: int, world_size: int, device: torch.device):
    if args.experiment == "baseline" and args.base_checkpoint:
        raise ValueError("The baseline cannot use --base-checkpoint")
    if args.experiment != "baseline" and not args.base_checkpoint and not (args.smoke_test or args.dt_sweep):
        raise ValueError("An ablation branch requires --base-checkpoint; run run_ablation.py for the complete workflow")
    if not (args.smoke_test or args.dt_sweep or args.eval_only):
        if args.warmup_epochs != 70 or args.device_epochs != 30 or args.extra_device_epochs != 0:
            raise ValueError("The specified ablation protocol requires exactly 70 baseline + 30 continuation epochs")
        if args.early_stop:
            raise ValueError("Early stopping is disabled: all three branches must train for exactly 30 epochs")
    if not torch.cuda.is_available():
        try:
            torch.set_num_threads(2 if world_size > 1 else 8)
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass

    cfg = Config(
        seed=args.seed,
        dataset_path=args.dataset,
        dt_s=args.dt_us * 1e-6,
        steps=16,  # intentionally fixed: train = val = test = 16
        warmup_epochs=args.warmup_epochs,
        device_epochs=args.device_epochs,
        extra_device_epochs=args.extra_device_epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        device=str(device),
    )
    seed_everything(cfg.seed + rank)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else SCRIPT_DIR
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("=" * 88)
        print(f"[LIF] algorithm dt = physical dt = {cfg.dt_s*1e6:.3f} us")
        print(f"[LIF] beta = exp(-dt/8.7us) = {beta_from_dt(cfg.dt_s):.9f}")
        print(f"[LIF] unified train/val/test steps = {cfg.steps}; window={cfg.steps*cfg.dt_s*1e6:.2f} us")
        print("[data] official OCTMNIST splits; exact train-image matches excluded from validation only")
        print(f"[training] experiment={args.experiment}; world_size={world_size}; global_batch_size={cfg.batch_size}")
        print(f"[target] {args.experiment} SNN: accuracy>=90%, every-class recall>=80%, loss<10")
        print("=" * 88)

    if world_size > 1 and rank != 0 and (args.dt_sweep or args.smoke_test or args.eval_only):
        return

    if args.dt_sweep:
        rows = dt_pilot_sweep(cfg, pilot_epochs=args.pilot_epochs)
        out = output_dir / "paper_dt_sweep_16steps.csv"
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["dt_us", "beta", "steps", "val_acc", "val_min_recall"])
            w.writerows(rows)
        print(f"[dt-sweep] saved -> {out.name}")
        return

    if args.smoke_test:
        smoke_test(cfg)
        return

    curve = DeviceCurve(find_input_file("line_LTP_LTD.csv", ["line_LTP_LTD*.csv"]), device)
    train_ds, val_ds, test_ds = load_full_splits(cfg)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, cfg, rank, world_size)
    model = OCTMemristorSNN(curve, cfg).to(device)
    train_model = (DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
                   if world_size > 1 and not args.eval_only else model)
    ckpt_path = output_dir / args.checkpoint
    report_ckpt_path = ckpt_path
    nparam = sum(p.numel() for p in model.parameters())
    training_history: list[dict] = []

    if rank == 0:
        print(f"[model] trainable parameters={nparam:,}; device={device}")
        print("[model] 2ch input / 48-48 / pool / 96-96 / pool / 192-192-192 / FC384-LIF / 4")
        print(
            f"[weights] trainable DeltaG is clamped to [{curve.delta_min:.6e},{curve.delta_max:.6e}] S; "
            "device mapping uses nearest measured differential state, not linear pulse index"
        )

    if args.eval_only:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.project_shadow_weights()
        if args.experiment == "measured":
            model.program_all_from_shadow()
            final_weight_mode = "programmed"
        elif args.experiment == "ideal":
            final_weight_mode = "ideal"
        else:
            final_weight_mode = "continuous"
        model.set_weight_mode(final_weight_mode)
        print(f"[checkpoint] loaded {ckpt_path.name} ({state.get('epoch_tag','unknown')})")
    else:
        # ------------------------------------------------------------------
        # Stage A: continuous SNN, but the actual trainable variable is DeltaG in siemens.
        # ------------------------------------------------------------------
        model.set_weight_mode("continuous")
        opt = build_optimizer(model, cfg.warmup_lr, cfg.weight_decay)
        # Lambda schedule preserves the relative LR scale between physical DeltaG parameters
        # (siemens) and the remaining trainable parameters.
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda ep: 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * ep / max(cfg.warmup_epochs, 1)))
        )
        best_shadow_score = -1.0
        shadow_path = (Path(args.base_checkpoint).expanduser().resolve() if args.base_checkpoint
                       else output_dir / "paper_continuous_shadow_best.pt")
        last_val = None

        for ep in range(1, cfg.warmup_epochs + 1) if not args.base_checkpoint else ():
            loss, tracc, rates = train_epoch(model, train_model, train_loader, opt, cfg, device,
                                             weight_mode="continuous", epoch=ep,
                                             rank=rank, world_size=world_size)
            if rank == 0:
                model.set_weight_mode("continuous")
                val = evaluate(model, val_loader, device, cfg.steps, tta_flip=False)
                last_val = val
                score = selection_score(val)
                print(
                    f"[continuous {ep:02d}] loss={loss:.4f} train={100*tracc:.2f}% "
                    f"val={100*val['acc']:.2f}% minR={100*val['min_recall']:.2f}% "
                    f"macroF1={100*val['macro_f1']:.2f}% rates={np.round(rates,3)}"
                )
                training_history.append(make_training_history_row(
                    stage="Stage_A_continuous_device_range_SNN",
                    weight_mode="ideal_float_shadow",
                    stage_epoch=ep, global_epoch=ep,
                    train_loss=loss, train_acc=tracc, val=val, score=score,
                    lr=max(pg["lr"] for pg in opt.param_groups), rates=rates, pulse_budget=0,
                ))
                if score > best_shadow_score:
                    best_shadow_score = score
                    save_ckpt(shadow_path, model, cfg, f"continuous-{ep}", val)
            sch.step()
            if world_size > 1:
                dist.barrier()

        if args.experiment == "baseline":
            if rank == 0:
                if last_val is None:
                    raise RuntimeError("The baseline trained no epochs")
                # Branches must start from the identical END-of-epoch-70 weights,
                # not from the best earlier validation epoch.
                base_path = output_dir / "base_epoch_70.pt"
                save_ckpt(base_path, model, cfg, f"continuous-{cfg.warmup_epochs}-last", last_val)
                save_training_and_lif_history(training_history, output_dir)
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                report_final_outputs(model, curve, cfg, args, output_dir, base_path,
                                     train_ds, val_ds, train_loader, val_loader, test_loader,
                                     device, "continuous")
            return

        if args.base_checkpoint:
            if not shadow_path.is_file():
                raise FileNotFoundError(f"Baseline checkpoint not found: {shadow_path}")
            baseline_history_path = shadow_path.parent / "paper_training_history_all.csv"
            if rank == 0:
                if not baseline_history_path.is_file():
                    raise FileNotFoundError(f"Baseline epoch history not found: {baseline_history_path}")
                with baseline_history_path.open(newline="", encoding="utf-8-sig") as stream:
                    training_history = list(csv.DictReader(stream))
                if len(training_history) != cfg.warmup_epochs:
                    raise ValueError("Baseline history length does not match --warmup-epochs")
                for row in training_history:
                    for name in LIF_LAYER_NAMES:
                        key = f"rate_{name}"
                        row[key] = float(row[key])

        shadow = torch.load(shadow_path, map_location=device)
        if args.base_checkpoint:
            saved_cfg = shadow.get("config", {})
            if (shadow.get("epoch_tag") != f"continuous-{cfg.warmup_epochs}-last"
                    or saved_cfg.get("seed") != cfg.seed
                    or saved_cfg.get("steps") != cfg.steps
                    or abs(float(saved_cfg.get("dt_s", -1)) - cfg.dt_s) > 1e-15):
                raise ValueError("Baseline checkpoint seed, epoch, steps or dt do not match this branch")
        model.load_state_dict(shadow["model"])
        model.project_shadow_weights()
        if rank == 0:
            print(f"[continuous] loaded shared final baseline {shadow['epoch_tag']}")

        if args.experiment == "float":
            model.set_weight_mode("continuous")
            # All branches restart an AdamW optimizer at epoch 71 with the same
            # continuation LR/schedule; only their forward weight map differs.
            opt = build_optimizer(model, cfg.device_lr, cfg.weight_decay * 0.5)
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda ep: 0.10 + 0.90 * 0.5 *
                (1.0 + math.cos(math.pi * ep / max(cfg.device_epochs, 1)))
            )
            best_score = -1.0
            targets_reached_on_val = False
            for dep in range(1, cfg.device_epochs + 1):
                loss, tracc, rates = train_epoch(
                    model, train_model, train_loader, opt, cfg, device,
                    weight_mode="continuous", epoch=cfg.warmup_epochs + dep,
                    rank=rank, world_size=world_size,
                )
                if rank == 0:
                    val = evaluate(model, val_loader, device, cfg.steps, tta_flip=False)
                    score = selection_score(val)
                    print(
                        f"[float continuation {dep:02d}] loss={loss:.4f} "
                        f"train={100*tracc:.2f}% val={100*val['acc']:.2f}% "
                        f"minR={100*val['min_recall']:.2f}% "
                        f"macroF1={100*val['macro_f1']:.2f}% score={score:.4f}"
                    )
                    training_history.append(make_training_history_row(
                        stage="Stage_B_float_continuation", weight_mode="ideal_float_shadow",
                        stage_epoch=dep, global_epoch=cfg.warmup_epochs + dep,
                        train_loss=loss, train_acc=tracc, val=val, score=score,
                        lr=max(pg["lr"] for pg in opt.param_groups), rates=rates, pulse_budget=0,
                    ))
                    if score > best_score + 1e-4:
                        best_score = score
                        save_ckpt(ckpt_path, model, cfg, f"float-continuation-{dep}", val)
                    if val["acc"] >= cfg.target_acc and val["min_recall"] >= cfg.target_min_recall:
                        targets_reached_on_val = True
                sch.step()
                if world_size > 1:
                    dist.barrier()
            if world_size > 1:
                dist.barrier()
            if not ckpt_path.exists():
                raise RuntimeError("No float-continuation checkpoint was saved")
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
            model.project_shadow_weights()
            final_weight_mode = "continuous"
        else:
            quantizer = "ideal" if args.experiment == "ideal" else "measured"
            qat_mode = "ideal_qat" if args.experiment == "ideal" else "qat"
            final_weight_mode = "ideal" if args.experiment == "ideal" else "programmed"
            history_mode = "ideal_uniform_quantized" if args.experiment == "ideal" else "mapped_physical_device"
            stage_name = "Stage_B_ideal_uniform_QAT" if args.experiment == "ideal" else "Stage_B_measured_state_QAT"

            if rank == 0:
                mean_qerr, max_qerr = model.quantization_stats(quantizer)
                print(
                    f"[{quantizer}-mapping] mean |DeltaGq-DeltaG|/range={mean_qerr:.3e}, "
                    f"max={max_qerr:.3e}"
                )
                model.set_weight_mode("ideal" if args.experiment == "ideal" else "qat")
                preqat = evaluate(model, val_loader, device, cfg.steps, tta_flip=False)
                print_metrics(f"VAL after {quantizer} quantization", preqat)
            if world_size > 1:
                dist.barrier()

            model.set_weight_mode(qat_mode)
            opt = build_optimizer(model, cfg.device_lr, cfg.weight_decay * 0.5)
            total_dev = cfg.device_epochs + cfg.extra_device_epochs
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda ep: 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * ep / max(total_dev, 1)))
            )
            best_score = -1.0
            stale = 0
            targets_reached_on_val = False

            for dep in range(1, total_dev + 1):
                loss, tracc, rates = train_epoch(
                    model, train_model, train_loader, opt, cfg, device,
                    weight_mode=qat_mode, epoch=cfg.warmup_epochs + dep,
                    rank=rank, world_size=world_size,
                )
                stop = torch.zeros(1, dtype=torch.int32, device=device)
                if rank == 0:
                    model.set_weight_mode(qat_mode)
                    val = evaluate(model, val_loader, device, cfg.steps, tta_flip=False)
                    score = selection_score(val)
                    reached = val["acc"] >= cfg.target_acc and val["min_recall"] >= cfg.target_min_recall
                    print(
                        f"[{args.experiment} QAT {dep:02d}] loss={loss:.4f} train={100*tracc:.2f}% "
                        f"val={100*val['acc']:.2f}% minR={100*val['min_recall']:.2f}% "
                        f"macroF1={100*val['macro_f1']:.2f}% score={score:.4f} rates={np.round(rates,3)}"
                    )
                    training_history.append(make_training_history_row(
                        stage=stage_name, weight_mode=history_mode,
                        stage_epoch=dep, global_epoch=cfg.warmup_epochs + dep,
                        train_loss=loss, train_acc=tracc, val=val, score=score,
                        lr=max(pg["lr"] for pg in opt.param_groups), rates=rates, pulse_budget=0,
                    ))
                    if score > best_score + 1e-4:
                        best_score = score
                        stale = 0
                        save_ckpt(ckpt_path, model, cfg, f"{args.experiment}-QAT-{dep}", val)
                    else:
                        stale += 1
                    if reached:
                        targets_reached_on_val = True
                    if args.early_stop:
                        if reached and stale >= 3:
                            stop.fill_(1)
                        elif dep >= cfg.device_epochs and stale >= cfg.device_patience:
                            stop.fill_(1)
                sch.step()
                if world_size > 1:
                    dist.broadcast(stop, src=0)
                if stop.item():
                    break

            if world_size > 1:
                dist.barrier()
            if not ckpt_path.exists():
                raise RuntimeError("No quantized checkpoint was saved")
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model"])
            model.project_shadow_weights()
            if args.experiment == "measured":
                model.program_all_from_shadow()
                if rank == 0:
                    programmed_state = dict(state)
                    programmed_state["model"] = model.state_dict()
                    programmed_state["epoch_tag"] = str(state.get("epoch_tag", "measured")) + "-programmed"
                    report_ckpt_path = output_dir / "programmed.pt"
                    torch.save(programmed_state, report_ckpt_path)
            model.set_weight_mode(final_weight_mode)

        if rank == 0:
            if not targets_reached_on_val:
                print("[warning] validation targets were not demonstrated; test labels were not used for tuning")
            save_training_and_lif_history(training_history, output_dir)
        if world_size > 1:
            dist.barrier()

    # ----------------------------------------------------------------------
    # PRIMARY FINAL RESULTS: raw, one-pass SNN for the selected weight condition, 16 steps.
    # ----------------------------------------------------------------------
    if world_size > 1 and rank != 0:
        return
    model.set_weight_mode(final_weight_mode)
    val_logits, val_y = collect_logits(model, val_loader, device, cfg.steps, tta_flip=False)
    val_final = metrics_from_logits(val_logits, val_y, calibration=None)
    print_metrics(f"VAL-final {args.experiment} 16-step", val_final)

    test_logits, test_y = collect_logits(model, test_loader, device, cfg.steps, tta_flip=False)
    test_final = metrics_from_logits(test_logits, test_y, calibration=None)
    print_metrics(f"TEST {args.experiment} 16-step", test_final)

    train_eval_loader = DataLoader(
        train_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=torch.cuda.is_available()
    )
    train_logits, train_y = collect_logits(model, train_eval_loader, device, cfg.steps, tta_flip=False)
    train_final = metrics_from_logits(train_logits, train_y, calibration=None)
    print_metrics(f"TRAIN-final {args.experiment} 16-step", train_final)

    save_confusion_matrix_csv(output_dir / "paper_confusion_matrix_train.csv", train_final)
    save_confusion_matrix_csv(output_dir / "paper_confusion_matrix_validation.csv", val_final)
    save_confusion_matrix_csv(output_dir / "paper_confusion_matrix_test.csv", test_final)
    save_final_split_metrics_csv(
        output_dir / "paper_final_split_metrics.csv",
        [("train", train_final), ("validation", val_final), ("test", test_final)]
    )

    if args.report_tta:
        tta = evaluate(model, test_loader, device, cfg.steps, tta_flip=True)
        print_metrics("TEST supplementary flip-TTA", tta)

    if args.experiment == "measured":
        p_ltp, p_ltd = model.pulse_counts()
        print(f"[device] canonical programming pulse events: LTP={p_ltp:,}, LTD={p_ltd:,}")

    passed = (
        test_final["acc"] >= cfg.target_acc and
        test_final["min_recall"] >= cfg.target_min_recall and
        test_final["loss"] < 10.0
    )
    print(
        f"[target] {args.experiment} accuracy>=90%, every-class recall>=80%, loss<10: "
        f"{'PASS' if passed else 'NOT REACHED'}"
    )

    out_meta = output_dir / "paper_final_metadata.pt"
    torch.save(dict(
        checkpoint=str(report_ckpt_path.name), experiment=args.experiment, config=asdict(cfg),
        test_acc=test_final["acc"], test_loss=test_final["loss"],
        test_recall=test_final["recall"], test_cm=test_final["cm"], passed=passed,
        device_g_min=curve.g_min, device_g_max=curve.g_max,
        differential_delta_min=curve.delta_min, differential_delta_max=curve.delta_max,
        ideal_uniform_levels=curve.ideal_levels,
        measured_ordered_pair_states=int(curve.diff_values.numel()),
    ), out_meta)

    if not args.no_plots:
        plot_device(curve, output_dir / "paper_device_curves.png")
        identity_calibration = (1.0, torch.zeros(N_CLASSES))
        plot_receptive_fields(model, val_ds, device, cfg, identity_calibration, output_dir / "new2.png")


def main():
    args = parse_args()
    rank, world_size, device = distributed_context(args.local_rank)
    try:
        run_main(args, rank, world_size, device)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
