import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from contextlib import nullcontext

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None, scaler=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            delta.requires_grad = True
            with torch.amp.autocast('cuda') if scaler is not None else nullcontext():
                output = model(X + delta)
                index = torch.where(output.max(1)[1] == y)
                if len(index[0]) == 0:
                    break
                loss = F.cross_entropy(output, y)
            if len(index[0]) == 0:
                continue
            if delta.grad is not None:
                delta.grad.zero_()
            if opt is not None and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta = delta.detach()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

def evaluate_fgsm(test_loader, model):
    epsilon = (8 / 255.) / std
    fgsm_loss = 0
    fgsm_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            delta = torch.zeros_like(X)
            delta.requires_grad = True
            with nullcontext():
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()
            grad = delta.grad.detach()
            # Broadcast epsilon to match channel shape if needed
            delta.data = clamp(epsilon * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta = delta.detach()  # Detach to avoid graph accumulation
            with torch.no_grad():
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                fgsm_loss += loss.item() * y.size(0)
                fgsm_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
    return fgsm_loss/n, fgsm_acc/n

def log_metrics(logfile, fieldnames, row):
    """Append a row of metrics to a CSV file."""
    file_exists = os.path.isfile(logfile)
    with open(logfile, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def plot_metrics(logfile, output_dir=None):
    """Plot learning rate, train loss, train acc, test acc, adversarial acc from a CSV log file."""
    import pandas as pd
    if not os.path.exists(logfile):
        print(f"Log file {logfile} does not exist.")
        return
    df = pd.read_csv(logfile)
    if output_dir is None:
        output_dir = os.path.dirname(logfile)
    # Plot learning rate
    if 'lr' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['lr'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
        plt.close()
    # Plot train loss
    if 'train_loss' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['train_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'train_loss.png'))
        plt.close()
    # Plot train acc
    if 'train_acc' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
        if 'test_acc' in df.columns:
            plt.plot(df['epoch'], df['test_acc'], label='Test Acc')
        if 'pgd_acc' in df.columns:
            plt.plot(df['epoch'], df['pgd_acc'], label='PGD Acc')
        if 'fgsm_acc' in df.columns:
            plt.plot(df['epoch'], df['fgsm_acc'], label='FGSM Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'accuracy.png'))
        plt.close()
    # Plot adversarial accuracy separately if needed
    if 'pgd_acc' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['pgd_acc'], label='PGD Acc')
        plt.xlabel('Epoch')
        plt.ylabel('PGD Accuracy')
        plt.title('PGD Adversarial Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'pgd_accuracy.png'))
        plt.close()
    if 'fgsm_acc' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['fgsm_acc'], label='FGSM Acc')
        plt.xlabel('Epoch')
        plt.ylabel('FGSM Accuracy')
        plt.title('FGSM Adversarial Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'fgsm_accuracy.png'))
        plt.close()
