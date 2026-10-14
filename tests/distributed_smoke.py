from pathlib import Path
import os
import socket
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def run():
    rank, world_size, device = main.distributed_context()
    try:
        torch.set_num_threads(2)
        main.seed_everything(19 + rank)
        source = Path(__file__).resolve().parents[1]
        curve = main.DeviceCurve(source / "line_LTP_LTD.csv", device)
        cfg = main.Config(batch_size=6, device=str(device))
        images = np.random.default_rng(0).integers(0, 256, size=(10, 28, 28), dtype=np.uint8)
        labels = np.arange(10) % main.N_CLASSES
        dataset = main.OCTDataset(images, labels)
        if world_size > 1:
            sampler = main.FullSplitBatchSampler(len(dataset), cfg.batch_size, rank, world_size, cfg.seed)
            loader = DataLoader(dataset, batch_sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        model = main.OCTMemristorSNN(curve, cfg).to(device)
        wrapped = (DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
                   if world_size > 1 else model)

        optimizer = main.build_optimizer(model, cfg.warmup_lr, cfg.weight_decay)
        loss, accuracy, _ = main.train_epoch(model, wrapped, loader, optimizer, cfg, device,
                                             qat=False, epoch=1, rank=rank, world_size=world_size)
        assert np.isfinite(loss) and 0 <= accuracy <= 1

        model.program_all_from_shadow()
        optimizer = main.build_optimizer(model, cfg.device_lr, cfg.weight_decay * 0.5)
        loss, accuracy, _ = main.train_epoch(model, wrapped, loader, optimizer, cfg, device,
                                             qat=True, epoch=2, rank=rank, world_size=world_size)
        assert np.isfinite(loss) and 0 <= accuracy <= 1

        checksums = [None] * world_size
        checksum = sum(float(p.detach().double().sum()) for p in model.parameters())
        if world_size > 1:
            dist.all_gather_object(checksums, checksum)
        else:
            checksums[0] = checksum
        if rank == 0:
            assert max(checksums) - min(checksums) < 1e-5, checksums
            print(f"PASS: two-stage training on {world_size} process(es) with synchronized parameters", flush=True)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def local_worker(rank: int, port: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["USE_LIBUV"] = "0"
    run()


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        run()
    else:
        world_size = int(os.environ.get("SPS_TEST_NPROCS", "2"))
        if world_size == 1:
            run()
        else:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            torch.multiprocessing.spawn(local_worker, args=(port, world_size), nprocs=world_size, join=True)
