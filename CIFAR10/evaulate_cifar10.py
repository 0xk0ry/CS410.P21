#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.models as models

def fgsm_attack(model, x, y, epsilon, device):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    outputs = model(x_adv)
    loss = nn.CrossEntropyLoss()(outputs, y.to(device))
    model.zero_grad(); loss.backward()
    grad = x_adv.grad.data
    x_adv = x_adv + epsilon * grad.sign()
    return torch.clamp(x_adv, 0, 1)


def fgsm_rs_attack(model, x, y, epsilon, alpha_init, device):
    delta = torch.empty_like(x).uniform_(-alpha_init, alpha_init).to(device)
    x0 = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
    outputs = model(x0)
    loss = nn.CrossEntropyLoss()(outputs, y.to(device))
    model.zero_grad(); loss.backward()
    grad = x0.grad.data
    x_adv = x0 + epsilon * grad.sign()
    return torch.clamp(x_adv, 0, 1)


def pgd_attack(model, x, y, epsilon, alpha, iters, device):
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon).to(device)
    for _ in range(iters):
        x_adv = torch.clamp(x + delta, 0, 1).detach().requires_grad_(True)
        outputs = model(x_adv)
        loss = nn.CrossEntropyLoss()(outputs, y.to(device))
        model.zero_grad(); loss.backward()
        grad = x_adv.grad.data
        delta = torch.clamp(delta + alpha * grad.sign(), -epsilon, epsilon)
    x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv


def evaluate(model, loader, device, attack, epsilon, alpha, alpha_init, iters):
    model.to(device).eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if attack == 'none':
            x_eval = x
        elif attack == 'fgsm':
            x_eval = fgsm_attack(model, x, y, epsilon, device)
        elif attack == 'fgsm_rs':
            x_eval = fgsm_rs_attack(model, x, y, epsilon, alpha_init, device)
        elif attack == 'pgd':
            x_eval = pgd_attack(model, x, y, epsilon, alpha, iters, device)
        else:
            raise ValueError(f"Unsupported attack: {attack}")
        with torch.no_grad():
            outputs = model(x_eval)
        _, preds = outputs.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = 100.0 * correct / total
    print(f"{attack.upper()} Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10 Model with Adversarial Attacks')
    parser.add_argument('--data-dir', type=str, required=True, help='CIFAR-10 data directory')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--attack', choices=['none','fgsm','fgsm_rs','pgd'], default='none')
    parser.add_argument('--epsilon', type=float, default=8.0, help='Attack budget')
    parser.add_argument('--alpha', type=float, default=2.0, help='Step size for PGD')
    parser.add_argument('--alpha-init', type=float, default=1.0, help='Random start radius for FGSM-RS')
    parser.add_argument('--attack-iters', type=int, default=7, help='Iterations for PGD')
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
    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    model = models.resnet18(num_classes=10)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    # Evaluate for chosen attack
    evaluate(model, loader, device,
             args.attack, args.epsilon,
             args.alpha, args.alpha_init,
             args.attack_iters)