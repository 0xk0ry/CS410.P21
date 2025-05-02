#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.models as models

def evaluate_clean(model, loader, device):
    model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    print(f"Clean Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10 Clean Accuracy')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to CIFAR-10 data directory')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--use-cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    # Data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    testset = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    # Model
    model = models.resnet18(num_classes=10)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    # Evaluate clean accuracy
    evaluate_clean(model, loader, device)
