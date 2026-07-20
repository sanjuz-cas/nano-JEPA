import datetime
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

from config import parse_config
from data import build_dataset, build_loader
from model import NanoJEPA, jepa_loss


# ---------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------
def setup_dist():
    # Kaggle T4 boxes often do better with P2P/IB disabled.
    # If you are on a workstation with NVLink, you can remove these.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=30),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True

    torch.cuda.set_device(0)
    return 0, 1, 0, False


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int):
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        model = model.module

    return getattr(model, "_orig_mod", model)


def adjust_lr(optimizer, step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for g in optimizer.param_groups:
        g["lr"] = lr

    return lr


def cosine_schedule(step: int, total_steps: int, base: float, end: float) -> float:
    if total_steps <= 1:
        return base

    progress = min(step / max(1, total_steps - 1), 1.0)
    return base + (end - base) * 0.5 * (1.0 - math.cos(math.pi * progress))


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, cfg):
    obj = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "args": vars(cfg) if hasattr(cfg, "__dict__") else {},
    }

    torch.save(obj, path)


# ---------------------------------------------------------------------
# Optional t-SNE export
# ---------------------------------------------------------------------
@torch.no_grad()
def run_tsne(model_without_ddp: NanoJEPA, cfg, device: torch.device):
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("scikit-learn and matplotlib are required for --tsne. Skipping t-SNE.")
        return

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = T.Compose(
        [
            T.Resize(cfg.img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(
        root=cfg.data_path,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    feats = []
    labels = []
    collected = 0

    model_without_ddp.eval()

    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(enabled=not cfg.disable_fp16):
            tokens = model_without_ddp.target_patch_embed(x).flatten(2).transpose(1, 2)
            tokens = tokens + model_without_ddp.target_pos_embed
            out = model_without_ddp.target_encoder(tokens)
            feat = out.mean(dim=1)

        feats.append(feat.float().cpu())
        labels.append(y)

        collected += feat.size(0)
        if collected >= cfg.tsne_samples:
            break

    feats = torch.cat(feats, dim=0)[: cfg.tsne_samples]
    labels = torch.cat(labels, dim=0)[: cfg.tsne_samples]

    print(f"Running t-SNE on {feats.shape[0]} samples...")
    emb = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(feats.numpy())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels.numpy(),
        cmap="tab10",
        s=8,
        alpha=0.85,
    )
    plt.colorbar(scatter)
    plt.title("Nano JEPA latent t-SNE")
    out_path = Path(cfg.output_dir) / "tsne_nano_jepa.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved t-SNE figure to {out_path}")


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train_one_epoch(
    model,
    model_without_ddp,
    loader,
    optimizer,
    scaler,
    epoch: int,
    cfg,
    device: torch.device,
    world_size: int,
    rank: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    total_params: int,
):
    model.train()
    model_without_ddp.target_encoder.eval()

    if loader.sampler is not None and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    rank0 = rank == 0
    window_images = 0
    window_start = None

    for step, (images, _) in enumerate(loader):
        if cfg.benchmark_steps > 0 and global_step >= cfg.benchmark_steps:
            break

        lr = adjust_lr(optimizer, global_step, total_steps, warmup_steps, base_lr)
        momentum = cosine_schedule(global_step, total_steps, cfg.momentum, cfg.momentum_end)

        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=not cfg.disable_fp16):
            pred, target = model(images)
            loss = jepa_loss(
                pred,
                target,
                std_weight=cfg.std_weight,
                cov_weight=cfg.cov_weight,
            )

        scaler.scale(loss).backward()

        if cfg.clip_grad > 0:
            if not cfg.disable_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)

        scaler.step(optimizer)
        scaler.update()

        model_without_ddp.update_target(momentum)

        batch_images = images.size(0) * world_size

        # Skip the first few warmup iterations for stable throughput measurement.
        if global_step >= 5:
            if window_start is None:
                window_start = time.perf_counter()
            window_images += batch_images

        if rank0 and window_images > 0 and (step + 1) % cfg.log_interval == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - window_start
            img_sec = window_images / max(elapsed, 1e-9)

            # Calibrated MFU using your reported Phase-1 accounting.
            tflops = img_sec * cfg.flops_per_image / 1e12
            mfu = 100.0 * tflops / cfg.peak_tflops

            # Theoretical transformer-style estimate for diagnostics only.
            theo_tflops = img_sec * 6.0 * total_params / 1e12
            theo_mfu = 100.0 * theo_tflops / cfg.peak_tflops

            print(
                f"[Epoch {epoch:03d}][Step {global_step:06d}] "
                f"loss={loss.item():.4f} lr={lr:.6f} mom={momentum:.5f} "
                f"img/sec={img_sec:.1f} "
                f"TFLOPS={tflops:.3f} MFU={mfu:.3f}% "
                f"theo_TFLOPS={theo_tflops:.3f} theo_MFU={theo_mfu:.3f}%",
                flush=True,
            )

            window_images = 0
            window_start = None

        global_step += 1

    if rank0 and window_images > 0 and window_start is not None:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - window_start
        img_sec = window_images / max(elapsed, 1e-9)
        tflops = img_sec * cfg.flops_per_image / 1e12
        mfu = 100.0 * tflops / cfg.peak_tflops

        print(
            f"[Epoch {epoch:03d}][Final] "
            f"img/sec={img_sec:.1f} TFLOPS={tflops:.3f} MFU={mfu:.3f}%",
            flush=True,
        )

    return global_step


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = parse_config()

    if cfg.disable_vicreg:
        cfg.std_weight = 0.0
        cfg.cov_weight = 0.0

    rank, world_size, local_rank, ddp = setup_dist()
    device = torch.device(f"cuda:{local_rank}")

    set_seed(cfg.seed, rank)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    if rank == 0:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    global_batch = cfg.batch_size * world_size
    base_lr = cfg.lr
    if cfg.scale_lr:
        base_lr = cfg.lr * float(global_batch) / float(cfg.lr_scale_base_batch)

    if rank == 0:
        print("=" * 80)
        print("Nano JEPA HPC Trainer")
        print("=" * 80)
        print(f"DDP enabled          : {ddp}")
        print(f"World size           : {world_size}")
        print(f"Per-GPU batch size   : {cfg.batch_size}")
        print(f"Global batch size    : {global_batch}")
        print(f"Base LR              : {base_lr}")
        print(f"FP16                 : {not cfg.disable_fp16}")
        print(f"Gradient checkpoint  : {not cfg.disable_grad_checkpointing}")
        print(f"torch.compile        : {cfg.compile}")
        print(f"MFU FLOPs/image      : {cfg.flops_per_image:.3e}")
        print(f"Peak TFLOPS          : {cfg.peak_tflops}")
        print("=" * 80, flush=True)

    model = NanoJEPA(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        context_depth=cfg.context_depth,
        target_depth=cfg.target_depth,
        predictor_depth=cfg.predictor_depth,
        heads=cfg.heads,
        mlp_ratio=cfg.mlp_ratio,
        mask_ratio=cfg.mask_ratio,
        grad_ckpt=not cfg.disable_grad_checkpointing,
    ).to(device)

    # Better Conv2d memory layout for T4.
    model.patch_embed.weight.data = model.patch_embed.weight.data.contiguous(memory_format=torch.channels_last)
    model.target_patch_embed.weight.data = model.target_patch_embed.weight.data.contiguous(memory_format=torch.channels_last)

    if cfg.compile:
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

        # Compile only the heavy trainable transformers, not the EMA target branch.
        model.context_encoder = torch.compile(model.context_encoder, mode=cfg.compile_mode)
        model.predictor = torch.compile(model.predictor, mode=cfg.compile_mode)

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            static_graph=cfg.static_graph,
            find_unused_parameters=False,
        )

    model_without_ddp = unwrap_model(model)

    total_params = sum(p.numel() for p in model_without_ddp.parameters())
    trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

    if rank == 0:
        print(f"Total parameters     : {total_params / 1e6:.2f}M")
        print(f"Trainable parameters : {trainable_params / 1e6:.2f}M", flush=True)

    # Optimizer
    no_decay_keys = ("bias", "norm", "pos_embed", "mask_token")
    decay_params = []
    no_decay_params = []

    for n, p in model_without_ddp.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in no_decay_keys):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=(0.9, 0.95),
            fused=True,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=(0.9, 0.95),
        )

    scaler = torch.cuda.amp.GradScaler(enabled=not cfg.disable_fp16)

    dataset, sampler = build_dataset(cfg, rank, world_size, ddp)
    loader = build_loader(dataset, sampler, cfg)

    total_steps = len(loader) * cfg.epochs
    if cfg.benchmark_steps > 0:
        total_steps = cfg.benchmark_steps

    warmup_steps = len(loader) * cfg.warmup_epochs
    if cfg.benchmark_steps > 0:
        warmup_steps = min(warmup_steps, max(1, total_steps // 5))

    start_epoch = 0

    if cfg.resume and Path(cfg.resume).is_file():
        if rank == 0:
            print(f"Resuming from {cfg.resume}")
        ckpt = torch.load(cfg.resume, map_location=device)
        model_without_ddp.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1

    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch, cfg.epochs):
        global_step = train_one_epoch(
            model=model,
            model_without_ddp=model_without_ddp,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            cfg=cfg,
            device=device,
            world_size=world_size,
            rank=rank,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
            total_params=total_params,
        )

        if cfg.benchmark_steps > 0 and global_step >= cfg.benchmark_steps:
            break

        if rank == 0 and cfg.benchmark_steps == 0 and cfg.save_every > 0:
            if (epoch + 1) % cfg.save_every == 0 or (epoch + 1) == cfg.epochs:
                ckpt_path = Path(cfg.output_dir) / f"nano_jepa_epoch_{epoch:03d}.pth"
                save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, cfg)
                print(f"Saved checkpoint: {ckpt_path}", flush=True)

        if ddp:
            dist.barrier()

    if cfg.tsne and rank == 0 and not cfg.synthetic and cfg.benchmark_steps == 0:
        run_tsne(model_without_ddp, cfg, device)

    cleanup_dist()


if __name__ == "__main__":
    main()