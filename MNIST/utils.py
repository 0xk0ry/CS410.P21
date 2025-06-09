import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# MNIST stats - grayscale image normalization
mnist_mean = (0.1307,)
mnist_std = (0.3081,)

# For MNIST, we use simpler bounds of [0,1] since we don't normalize


def clamp(X, lower_limit, upper_limit):
    """
    Clamp tensor X between lower_limit and upper_limit.
    Both limits can be either tensors of same shape as X or scalars.
    """
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None, scaler=None):
    """
    PGD attack optimized for MNIST as per evaluate_mnist.py
    """
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        delta.data = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            d = clamp(d, 0-X, 1-X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    # For MNIST, we use scalar epsilon and alpha
    epsilon = 0.3  # Standard for MNIST
    alpha = 0.01   # Step size for MNIST PGD
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
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

def attack_fgsm(model, X, y, epsilon):
    """
    FGSM attack implementation for MNIST as per evaluate_mnist.py
    """
    delta = torch.zeros_like(X, requires_grad=True)
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    return delta.detach()

def evaluate_fgsm(test_loader, model):
    # For MNIST, we use scalar epsilon
    epsilon = 0.3  # Standard for MNIST
    fgsm_loss = 0
    fgsm_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        delta = attack_fgsm(model, X, y, epsilon)
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

def plot_metrics(logfile, model_name, output_dir=None):
    """Plot learning rate, train loss, train acc, test acc, adversarial acc from a CSV log file."""
    import pandas as pd
    if not os.path.exists(logfile):
        print(f"Log file {logfile} does not exist.")
        return
    df = pd.read_csv(logfile)
    if output_dir is None:
        output_dir = os.path.dirname(logfile)    # Plot learning rate
    if 'lr' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['lr'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.xticks(range(0, int(max(df['epoch'])) + 2, 2))  # Set x-ticks every 2 epochs
        plt.savefig(os.path.join(output_dir, f'{model_name}_learning_rate.png'))
        plt.close()
    # Plot train loss
    if 'train_loss' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['train_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.xticks(range(0, int(max(df['epoch'])) + 2, 2))  # Set x-ticks every 2 epochs
        plt.savefig(os.path.join(output_dir, f'{model_name}_train_loss.png'))
        plt.close()    # Plot train acc
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
        plt.xticks(range(0, int(max(df['epoch'])) + 2, 2))  # Set x-ticks every 2 epochs
        plt.savefig(os.path.join(output_dir, f'{model_name}_accuracy.png'))
        plt.close()
    # Plot adversarial accuracy separately if needed
    if 'pgd_acc' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['pgd_acc'], label='PGD Acc')
        plt.xlabel('Epoch')
        plt.ylabel('PGD Accuracy')
        plt.title('PGD Adversarial Accuracy')
        plt.grid(True)
        plt.xticks(range(0, int(max(df['epoch'])) + 2, 2))  # Set x-ticks every 2 epochs
        plt.savefig(os.path.join(output_dir, f'{model_name}_pgd_accuracy.png'))
        plt.close()
    if 'fgsm_acc' in df.columns:
        plt.figure()
        plt.plot(df['epoch'], df['fgsm_acc'], label='FGSM Acc')
        plt.xlabel('Epoch')
        plt.ylabel('FGSM Accuracy')
        plt.title('FGSM Adversarial Accuracy')
        plt.grid(True)
        plt.xticks(range(0, int(max(df['epoch'])) + 2, 2))  # Set x-ticks every 2 epochs
        plt.savefig(os.path.join(output_dir, f'{model_name}_fgsm_accuracy.png'))
        plt.close()
