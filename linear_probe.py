import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from model import NanoJEPA

class JEPABackbone(nn.Module):
    """Wraps the pre-trained JEPA context encoder for feature extraction."""
    def __init__(self, jepa_model):
        super().__init__()
        self.patch_embed = jepa_model.patch_embed
        self.pos_embed = jepa_model.pos_embed
        self.context_encoder = jepa_model.context_encoder
        
    def forward(self, x):
        # Extract patches and add positional embeddings
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        
        # Pass ALL tokens through the context encoder (no masking for inference)
        out = self.context_encoder(tokens)
        
        # Global average pooling over the patch dimension to get image-level embedding
        return out.mean(dim=1)


class LinearProbe(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.bn = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Backbone is frozen, so we don't need to track gradients for it
        with torch.no_grad():
            feats = self.backbone(x)
        feats = self.bn(feats)
        return self.head(feats)

def get_transforms(img_size):
    # ImageNet stats (same as train.py)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    train_transform = T.Compose([
        T.Resize(int(img_size * 1.125)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    val_transform = T.Compose([
        T.Resize(int(img_size * 1.125)),
        T.CenterCrop(img_size), # Crucial: forces perfect square for pos_embed
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    return train_transform, val_transform

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.head.train() # Only the linear head trains
    model.backbone.eval() # Backbone stays frozen in eval mode
    
    running_loss, correct, total = 0.0, 0, 0
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_time = time.time() - start_time
    train_loss = running_loss / total
    train_acc = 100. * correct / total
    print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Time: {epoch_time:.1f}s")
    return train_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="JEPA Linear Probing Evaluation")
    
    # Paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained NanoJEPA .pth file")
    parser.add_argument("--train-path", type=str, required=True, help="Path to TRAIN ImageFolder")
    parser.add_argument("--val-path", type=str, required=True, help="Path to TEST/VAL ImageFolder")
    parser.add_argument("--output-dir", type=str, default="./outputs_probe")
    
    # Architecture (Must match pre-trained model)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--context-depth", type=int, default=12)
    parser.add_argument("--target-depth", type=int, default=12)
    parser.add_argument("--predictor-depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=12)
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("JEPA Linear Probing Evaluation")
    print("="*60)
    
    jepa = NanoJEPA(
        img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
        context_depth=args.context_depth, target_depth=args.target_depth, 
        predictor_depth=args.predictor_depth, heads=args.heads, grad_ckpt=False
    ).to(device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model']
    
    # Clean torch.compile prefixes if they exist
    cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    jepa.load_state_dict(cleaned_state_dict)
    
    backbone = JEPABackbone(jepa).to(device)
    for param in backbone.parameters():
        param.requires_grad = False
        
    train_dataset = torchvision.datasets.ImageFolder(args.train_path)
    num_classes = len(train_dataset.classes)
    print(f"Detected {num_classes} classes: {train_dataset.classes}")
    
    model = LinearProbe(backbone, args.embed_dim, num_classes).to(device)
    model = model.to(memory_format=torch.channels_last)
    
    train_transform, val_transform = get_transforms(args.img_size)
    
    train_dataset = torchvision.datasets.ImageFolder(args.train_path, transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(args.val_path, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)
    
    optimizer = optim.SGD(model.head.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"Trainable parameters: {sum(p.numel() for p in model.head.parameters()) / 1e3:.2f}K (Linear Head Only)")
    print(f"Starting Linear Probing for {args.epochs} epochs...\n")
    
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"[Epoch {epoch:03d}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Path(args.output_dir) / "best_linear_probe.pth")
            print(f"  -> Saved new best model with {best_acc:.2f}% accuracy!\n")
        else:
            print()
            
    print("="*60)
    print(f"TRAINING COMPLETE. Best Validation Accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
