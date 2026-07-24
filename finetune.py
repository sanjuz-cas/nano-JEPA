import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from model import NanoJEPA

class JEPAFineTune(nn.Module):
    def __init__(self, jepa_model, num_classes):
        super().__init__()
        self.patch_embed = jepa_model.patch_embed
        self.pos_embed = jepa_model.pos_embed
        self.encoder = jepa_model.context_encoder
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.norm.normalized_shape[0]),
            nn.Linear(self.encoder.norm.normalized_shape[0], num_classes)
        )
    
    def forward(self, x):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        out = self.encoder(tokens)
        return self.head(out.mean(dim=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--val-path", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--context-depth", type=int, default=6)
    parser.add_argument("--target-depth", type=int, default=6)
    parser.add_argument("--predictor-depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    train_tf = T.Compose([T.RandomResizedCrop(args.img_size, scale=(0.8,1.0)),
                          T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                          T.RandomRotation(180), T.ToTensor(), T.Normalize(mean, std)])
    val_tf = T.Compose([T.Resize(args.img_size), T.CenterCrop(args.img_size),
                        T.ToTensor(), T.Normalize(mean, std)])

    train_ds = torchvision.datasets.ImageFolder(args.train_path, transform=train_tf)
    val_ds = torchvision.datasets.ImageFolder(args.val_path, transform=val_tf)
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)

    # Load pretrained JEPA
    jepa = NanoJEPA(img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
                    context_depth=args.context_depth, target_depth=args.target_depth,
                    predictor_depth=args.predictor_depth, grad_ckpt=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    jepa.load_state_dict(cleaned)

    # Wrap for fine-tuning (entire backbone is trainable)
    model = JEPAFineTune(jepa, num_classes).to(device)
    model = model.to(memory_format=torch.channels_last)

    # Use a smaller LR for the backbone, larger for the head
    optimizer = optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "head" not in n], "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr * 10},
    ], weight_decay=0.05)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Fine-tuning on {num_classes} classes: {train_ds.classes}")
    print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        scheduler.step()
        train_acc = 100. * correct / total

        # Validate
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total += labels.size(0)
        
        val_acc = 100. * v_correct / v_total
        marker = " ★ BEST" if val_acc > best_acc else ""
        if val_acc > best_acc:
            best_acc = val_acc
        print(f"[Epoch {epoch:03d}] Train: {train_acc:.2f}% | Val: {val_acc:.2f}%{marker}")

    print(f"\n{'='*50}")
    print(f"BEST VALIDATION ACCURACY: {best_acc:.2f}%")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
