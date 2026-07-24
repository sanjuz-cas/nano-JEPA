import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from model import NanoJEPA

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        tokens = model.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + model.pos_embed
        out = model.context_encoder(tokens)
        feat = F.normalize(out.mean(dim=1), dim=1)  # L2 normalize!
        
        features.append(feat.cpu())
        labels.append(targets)
    
    return torch.cat(features), torch.cat(labels)

def knn_classifier(train_features, train_labels, test_features, test_labels, k=20, temperature=0.07):
    """Weighted k-NN classification"""
    # Compute similarity matrix
    similarity = torch.mm(test_features, train_features.T) / temperature
    
    # Get top-k nearest neighbors
    topk_sim, topk_idx = similarity.topk(k, dim=1)
    topk_labels = train_labels[topk_idx]
    
    # Weighted vote
    weights = torch.exp(topk_sim)
    
    # Count votes per class
    num_classes = train_labels.max().item() + 1
    votes = torch.zeros(len(test_labels), num_classes)
    for i in range(k):
        votes.scatter_add_(1, topk_labels[:, i:i+1], weights[:, i:i+1])
    
    predictions = votes.argmax(dim=1)
    accuracy = (predictions == test_labels).float().mean().item() * 100
    return accuracy

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
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = T.Compose([T.Resize(args.img_size), T.CenterCrop(args.img_size), 
                           T.ToTensor(), T.Normalize(mean, std)])

    # Load model
    model = NanoJEPA(img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
                     context_depth=args.context_depth, target_depth=args.target_depth,
                     predictor_depth=args.predictor_depth, grad_ckpt=False).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(cleaned)
    model = model.to(memory_format=torch.channels_last)

    # Extract features
    train_loader = DataLoader(torchvision.datasets.ImageFolder(args.train_path, transform=transform),
                              batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(torchvision.datasets.ImageFolder(args.val_path, transform=transform),
                            batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    print("Extracting training features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    print("Extracting test features...")
    test_features, test_labels = extract_features(model, val_loader, device)

    # Evaluate with different k values
    print("\n" + "="*50)
    print("k-NN Evaluation (Zero-Shot, No Training)")
    print("="*50)
    for k in [1, 5, 10, 20, 50]:
        acc = knn_classifier(train_features, train_labels, test_features, test_labels, k=k)
        print(f"  k={k:3d} | Accuracy: {acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
