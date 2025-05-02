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
    parser.add_argument('--attack', choices=['fgsm','fgsm_rs','pgd','free'], default='fgsm_rs',
                        help='Adversarial training method')
    parser.add_argument('--eps', type=float, default=2/255,
                        help='Max perturbation (e.g. 2/255)')
    parser.add_argument('--alpha_init', type=float, default=1/255,
                        help='Random start magnitude for FGSM-RS')
    parser.add_argument('--alpha', type=float, default=4/255,
                        help='Step size for PGD and Free')
    parser.add_argument('--num_steps', type=int, default=7,
                        help='Number of PGD iterations')
    parser.add_argument('--free_replays', type=int, default=4,
                        help='Number of replays for Free adversarial training')
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


def fgsm_rs_step(model, x, y, eps, alpha_init, device):
    delta = torch.empty_like(x).uniform_(-alpha_init, alpha_init).to(device)
    x0 = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
    outputs = model(x0)
    loss = nn.CrossEntropyLoss()(outputs, y)
    model.zero_grad(); loss.backward()
    grad = x0.grad.data
    return torch.clamp(x0 + eps * grad.sign(), 0, 1)


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

# Free adversarial: K-step FGSM replays

def free_step(model, x, y, eps, alpha, replays, device):
    delta = torch.zeros_like(x).to(device)
    for _ in range(replays):
        x_adv = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)
        model.zero_grad(); loss.backward()
        grad = x_adv.grad.data
        delta = torch.clamp(delta + alpha * grad.sign(), -eps, eps)
        # update model with same loss
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return torch.clamp(x + delta, 0, 1)

# ----------------------------
# Training one epoch
# ----------------------------
def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if args.attack == 'fgsm':
            x_adv = fgsm_step(model, x, y, args.eps, device)
        elif args.attack == 'fgsm_rs':
            x_adv = fgsm_rs_step(model, x, y, args.eps, args.alpha_init, device)
        elif args.attack == 'pgd':
            x_adv = pgd_step(model, x, y, args.eps, args.alpha, args.num_steps, device)
        elif args.attack == 'free':
            x_adv = free_step(model, x, y, args.eps, args.alpha, args.free_replays, device)
        else:
            raise ValueError(f"Unknown attack {args.attack}")
        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    # Model instantiation as in original
    if args.pretrained:
        print(f"=> using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()
    model.cuda()

    # parameter groups
    param_to_module = {}
    for module in model.modules():
        for p in module.parameters(recurse=False):
            param_to_module[p] = type(module).__name__
    group_decay    = [p for p in model.parameters() if 'BatchNorm' not in param_to_module[p]]
    group_no_decay = [p for p in model.parameters() if 'BatchNorm' in param_to_module[p]]
    optimizer = optim.SGD(
        [ {'params': group_decay}, {'params': group_no_decay, 'weight_decay': 0} ],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # mixed precision
    if args.half:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    model = nn.DataParallel(model)

    # Data
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    # Train
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        train_one_epoch(model, train_loader, optimizer, device, args)
        ckpt = os.path.join(args.out_dir, f"{args.attack}_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Epoch {epoch} done, saved to {ckpt}")

    # Final clean eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item(); total += y.size(0)
    print(f"Final Clean Acc: {100*correct/total:.2f}%")

if __name__ == '__main__':
    main()
