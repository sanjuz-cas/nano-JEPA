import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as T

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


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
        return T.Compose([
            T.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),         
            T.RandomRotation(180),          
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    return T.Compose([
        T.Resize(int(cfg.img_size * 1.125)),
        T.CenterCrop(cfg.img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_dataset(cfg, rank: int, world_size: int, ddp: bool):
    if cfg.synthetic:
        dataset = SyntheticDataset(size=getattr(cfg, 'synthetic_size', 100000), img_size=cfg.img_size)
        
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

    if ddp:
        dist.barrier()

    dataset = torchvision.datasets.ImageFolder(
        root=cfg.data_path,
        transform=transform,
    )

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



def fast_collate(batch):
    """
    Converts tensors to channels_last memory format on the CPU.
    This maximizes T4 Tensor Core utilization for the initial PatchEmbed Conv2d.
    """
    imgs = torch.utils.data.default_collate([b[0] for b in batch])
    labels = torch.utils.data.default_collate([b[1] for b in batch])
    
    # Convert to channels_last memory format
    imgs = imgs.to(memory_format=torch.channels_last)
    return imgs, labels


def build_loader(dataset, sampler, cfg, is_train=True):
    num_workers = min(getattr(cfg, 'workers', 4), 4)
    
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        pin_memory=True,               # Fast Host-to-Device VRAM transfer
        drop_last=is_train,
        collate_fn=fast_collate if is_train else None, # Apply channels_last optimization
    )

    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    else:
        loader_kwargs["shuffle"] = is_train

    if num_workers > 0 and is_train:
        loader_kwargs["persistent_workers"] = True   # Prevents CPU spin-up/down overhead
        loader_kwargs["prefetch_factor"] = getattr(cfg, 'prefetch', 2) # Pre-fetches batches to GPU

    return DataLoader(dataset, **loader_kwargs)


def build_tsne_loader(cfg):
    transform = T.Compose(
        [
            T.Resize(cfg.img_size),
            T.CenterCrop(cfg.img_size),
            T.ToTensor(),
            T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
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
