#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from apex import amp

# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='ImageNet Adversarial Training')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to ImageNet train/val folders')
    parser.add_argument('--arch', type=str, default='resnet50',
                        choices=['resnet50','resnet18'], help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--attack', choices=['fgsm','pgd','none'], default='none',
                        help='Attack type for evaluation')
    parser.add_argument('--eps', type=float, default=2/255,
                        help='Max perturbation (e.g. 2/255)')
    parser.add_argument('--alpha', type=float, default=4/255,
                        help='Step size for PGD')
    parser.add_argument('--num-steps', type=int, default=7,
                        help='Number of PGD iterations')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--half', action='store_true',
                        help='Use mixed precision (apex)')
    parser.add_argument('--out-dir', type=str, default='./outputs',
                        help='Directory to save checkpoints')
    parser.add_argument('--use-cuda', action='store_true')
    return parser.parse_args()

# ----------------------------
# Data loaders
# ----------------------------
def get_dataloaders(data_dir, batch_size):
    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds   = torchvision.datasets.ImageFolder(val_dir,   transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader

# ----------------------------
# Attack implementations
# ----------------------------
def fgsm_step(model, x, y, eps, device):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    outputs = model(x_adv)
    loss = nn.CrossEntropyLoss()(outputs, y)
    model.zero_grad(); loss.backward()
    grad = x_adv.grad.data
    return torch.clamp(x_adv + eps * grad.sign(), 0, 1)


def pgd_step(model, x, y, eps, alpha, steps, device):
    delta = torch.empty_like(x).uniform_(-eps, eps).to(device)
    for _ in range(steps):
        x_adv = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)
        model.zero_grad(); loss.backward()
        grad = x_adv.grad.data
        delta = torch.clamp(delta + alpha * grad.sign(), -eps, eps)
    return torch.clamp(x + delta, 0, 1)

# ----------------------------
# Training one epoch
# ----------------------------
def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # default no attack
        x_adv = x
        if args.attack == 'fgsm':
            x_adv = fgsm_step(model, x, y, args.eps, device)
        elif args.attack == 'pgd':
            x_adv = pgd_step(model, x, y, args.eps, args.alpha, args.num_steps, device)
        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# ----------------------------
# Evaluation
# ----------------------------
def evaluate(model, loader, device, attack, eps, alpha, steps):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if attack == 'fgsm':
                x_eval = fgsm_step(model, x, y, eps, device)
            elif attack == 'pgd':
                x_eval = pgd_step(model, x, y, eps, alpha, steps, device)
            else:
                x_eval = x
            preds = model(x_eval).argmax(dim=1)
            correct += (preds == y).sum().item(); total += y.size(0)
    acc = 100.0 * correct / total
    print(f"{attack.upper() if attack!='none' else 'CLEAN'} Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    # Model instantiation
    if args.pretrained:
        print(f"=> using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()
    model = model.cuda()

    # parameter groups
    param_to_module = {}
    for module in model.modules():
        for p in module.parameters(recurse=False):
            param_to_module[p] = type(module).__name__
    group_decay    = [p for p in model.parameters() if 'BatchNorm' not in param_to_module[p]]
    group_no_decay = [p for p in model.parameters() if 'BatchNorm'     in param_to_module[p]]
    optimizer = optim.SGD(
        [{'params': group_decay}, {'params': group_no_decay, 'weight_decay': 0}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    # mixed precision
    if args.half:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = nn.DataParallel(model)

    # Data loaders
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Training
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        train_one_epoch(model, train_loader, optimizer, device, args)
        ckpt = os.path.join(args.out_dir, f"{args.attack}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Epoch {epoch} done, saved to {ckpt}")

    # Final evaluation under all attacks
    for at in ['none', 'fgsm', 'pgd']:
        evaluate(model, val_loader, device, at, args.eps, args.alpha, args.num_steps)

if __name__ == '__main__':
    main()
