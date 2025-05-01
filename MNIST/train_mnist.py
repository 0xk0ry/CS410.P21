#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from mnist_net import mnist_net


def train(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dataset and loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    # Model and optimizer
    model = mnist_net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            # Free adversarial training
            if args.attack == 'free':
                delta = torch.zeros_like(x).to(device)
                for _ in range(args.free_replays):
                    # perturbed input
                    x_adv = torch.clamp(x + delta, 0, 1).requires_grad_(True)
                    outputs = model(x_adv)
                    loss = criterion(outputs, y)
                    optimizer.zero_grad()
                    loss.backward()
                    # update model
                    optimizer.step()
                    # update perturbation
                    grad = x_adv.grad.data
                    delta = torch.clamp(delta + args.alpha * grad.sign(), -args.epsilon, args.epsilon)
                continue

            # FGSM and FGSM-RS share most logic
            if args.attack in ['fgsm', 'fgsm_rs']:
                # initialization of delta
                if args.attack == 'fgsm':
                    if args.zero_init:
                        delta = torch.zeros_like(x)
                    else:
                        delta = torch.empty_like(x).uniform_(-args.alpha_init, args.alpha_init)
                else:  # fgsm_rs
                    delta = torch.empty_like(x).uniform_(-args.alpha_init, args.alpha_init)
                # random start
                x0 = torch.clamp(x + delta, 0, 1).requires_grad_(True)
                # single FGSM step
                outputs = model(x0)
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                grad = x0.grad.data
                delta = args.epsilon * grad.sign()
                x_adv = torch.clamp(x0 + delta, 0, 1)
                # update model on adversarial example
                outputs_adv = model(x_adv)
                loss_adv = criterion(outputs_adv, y)
                optimizer.zero_grad()
                loss_adv.backward()
                optimizer.step()
                continue

            # PGD adversarial training
            if args.attack == 'pgd':
                # random start in epsilon-ball
                delta = torch.empty_like(x).uniform_(-args.epsilon, args.epsilon).to(device)
                for _ in range(args.attack_iters):
                    x_delta = torch.clamp(x + delta, 0, 1).requires_grad_(True)
                    outputs = model(x_delta)
                    loss = criterion(outputs, y)
                    optimizer.zero_grad()
                    loss.backward()
                    grad = x_delta.grad.data
                    delta = torch.clamp(delta + args.alpha * grad.sign(), -args.epsilon, args.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
                optimizer.zero_grad()
                outputs_adv = model(x_adv)
                loss_adv = criterion(outputs_adv, y)
                loss_adv.backward()
                optimizer.step()
                continue

        print(f"Epoch {epoch}/{args.epochs} completed.")

    # Save model
    init_type = 'zero' if args.zero_init else 'random'
    torch.save(model.state_dict(), f"mnist_{args.attack}_{init_type}.pth")
    print("Training complete. Model saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Adversarial Training')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--attack', choices=['fgsm','fgsm_rs','pgd','free'], default='fgsm_rs',
                        help='Attack method: fgsm (one-step), fgsm_rs (fast random-step), pgd (iterative), free (free adversarial)')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Max perturbation')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Step size for PGD or Free replay')
    parser.add_argument('--alpha_init', type=float, default=0.3,
                        help='Random start magnitude for FGSM-RS or initial alpha for FGSM')
    parser.add_argument('--attack-iters', type=int, default=40, help='Number of PGD iterations')
    parser.add_argument('--free-replays', type=int, default=8,
                        help='Number of replays for Free adversarial training')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--zero-init', action='store_true', help='Use zero initialization for FGSM')
    args = parser.parse_args()
    train(args)
