import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as T


NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZE_STD = (0.2470, 0.2435, 0.2616)


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for pure GPU/MFU benchmarking.
    """

    def __init__(self, size: int = 100000, img_size: int = 64):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.rand(3, self.img_size, self.img_size, dtype=torch.float32)
        return x, 0


def build_transform(cfg):
    mean = NORMALIZE_MEAN
    std = NORMALIZE_STD

    if cfg.aug:
        return T.Compose(
            [
                T.Resize(int(cfg.img_size * 1.125)),
                T.RandomCrop(cfg.img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )

    return T.Compose(
        [
            T.Resize(cfg.img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def build_dataset(cfg, rank: int, world_size: int, ddp: bool):
    if cfg.synthetic:
        dataset = SyntheticDataset(size=cfg.synthetic_size, img_size=cfg.img_size)

        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            if ddp
            else None
        )

        return dataset, sampler

    transform = build_transform(cfg)

    # Use ImageFolder to read any folder-based dataset (e.g. blood-cells).
    if ddp:
        dist.barrier()

        dataset = torchvision.datasets.ImageFolder(
            root=cfg.data_path,
            transform=transform,
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    else:
        dataset = torchvision.datasets.ImageFolder(
            root=cfg.data_path,
            transform=transform,
        )

        sampler = None

    return dataset, sampler


def build_loader(dataset, sampler, cfg):
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )

    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = True

    if cfg.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = cfg.prefetch

    return DataLoader(dataset, **loader_kwargs)


def build_tsne_loader(cfg):
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

    return DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )