import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_net import mnist_net
import os
import csv
from utils import evaluate_standard, evaluate_pgd, evaluate_fgsm, log_metrics, plot_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['none', 'pgd', 'fgsm', 'free'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--alpha', default=0.375, type=float)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--free-replays', default=8, type=int)
    parser.add_argument('--lr-max', default=5e-3, type=float)
    parser.add_argument('--lr-type', default='flat', type=str, choices=['cyclic', 'flat'])
    parser.add_argument('--fname', default='mnist_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--zero-init', action='store_true', help='Use zero initialization for FGSM (no random start)')
    parser.add_argument('--out-dir', default='train_mnist_output', type=str, help='Output directory for logs and plots')
    return parser.parse_args()

def main():
    args = get_args()
    logger.info(args)
    print(args)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    logfile = os.path.join(args.out_dir, f'train_{args.fname}_output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    # Configure logging to file
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    # Set up CSV file for metrics
    csv_logfile = os.path.join(args.out_dir, f'train_{args.attack}_metrics.csv')
    fieldnames = ['epoch', 'lr', 'train_loss', 'train_acc', 'test_acc', 'pgd_acc', 'fgsm_acc']

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    
    # Add test dataset for evaluation
    mnist_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)

    model = mnist_net().cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    if args.lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat': 
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError('Unknown lr_type')

    criterion = nn.CrossEntropyLoss()

    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    print('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'fgsm':
                if args.zero_init:
                    delta = torch.zeros_like(X).cuda()
                else:
                    delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon).cuda()
                delta.requires_grad = True
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                delta = delta.detach()
                adv_X = torch.clamp(X + delta, 0, 1)
            elif args.attack == 'none':
                adv_X = X
            elif args.attack == 'pgd':
                delta = torch.zeros_like(X).uniform_(-args.epsilon, args.epsilon).cuda()
                delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                for _ in range(args.attack_iters):
                    delta.requires_grad = True
                    output = model(X + delta)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    I = output.max(1)[1] == y
                    delta.data[I] = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)[I]
                    delta.data[I] = torch.max(torch.min(1-X, delta.data), 0-X)[I]
                delta = delta.detach()
                adv_X = torch.clamp(X + delta, 0, 1)
            elif args.attack == 'free':
                # Free adversarial training
                delta = torch.zeros_like(X).cuda()
                for _ in range(args.free_replays):
                    delta.requires_grad = True
                    output = model(torch.clamp(X + delta, 0, 1))
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = torch.clamp(delta + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)
                    delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
                    delta = delta.detach()
                    # update model
                    output = model(torch.clamp(X + delta, 0, 1))
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    # accumulate metrics for each replay
                    train_loss += loss.item() * y.size(0)
                    train_acc += (output.max(1)[1] == y).sum().item()
                    train_n += y.size(0)
                continue  # skip the rest of the loop for free
            else:
                raise ValueError('Unknown attack')

            output = model(adv_X)
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        # Evaluate on test set and adversarial test examples after each epoch
        test_loss, test_acc = evaluate_standard(test_loader, model)
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 10, 1)
        fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model)
        
        # Log metrics to CSV
        log_metrics(csv_logfile, fieldnames, {
            'epoch': epoch,
            'lr': lr,
            'train_loss': train_loss/train_n,
            'train_acc': train_acc/train_n,
            'test_acc': test_acc,
            'pgd_acc': pgd_acc,
            'fgsm_acc': fgsm_acc
        })

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
        print('%d \t %.1f \t %.4f \t %.4f \t %.4f' % (
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n))
        torch.save(model.state_dict(), os.path.join(args.out_dir, args.fname+'.pth'))
    
    # Total training time
    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
    print('Total train time: %.4f minutes' % ((train_time - start_train_time)/60))
    
    # Plot all metrics at the end
    plot_metrics(csv_logfile, args.out_dir)

    # Final Evaluation
    model_test = mnist_net().cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t FGSM Loss \t FGSM Acc')
    print('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t FGSM Loss \t FGSM Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc, fgsm_loss, fgsm_acc)
    print('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' % (test_loss, test_acc, pgd_loss, pgd_acc, fgsm_loss, fgsm_acc))

if __name__ == "__main__":
    main()