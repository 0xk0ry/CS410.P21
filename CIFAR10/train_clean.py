import argparse
import copy
import logging
import os
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.amp import GradScaler, autocast

from preact_resnet import PreActResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
                   attack_pgd, evaluate_pgd, evaluate_standard, evaluate_fgsm, log_metrics, plot_metrics)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic',
                        choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
                        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_fgsm_output',
                        type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true',
                        help='Early stop if overfitting occurs')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
                        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
                        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
                        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--eval-freq', default=1, type=int,
                        help='Frequency of evaluation during training (epochs)')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    logfile = os.path.join(
        args.out_dir, f'cifar10_fgsm_{args.delta_init}.log')

    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()  # This sends output to console
        ])
    logger.info(args)
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    model = PreActResNet18().cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level,
                    loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    scaler = GradScaler()

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        
    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    csv_logfile = os.path.join(
        args.out_dir, f'cifar10_clean_metrics.csv')
    fieldnames = ['epoch', 'lr', 'train_loss',
                  'train_acc', 'test_acc', 'pgd_acc', 'fgsm_acc']

    # Training
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    print('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

    batch_scheduler = args.lr_schedule == 'cyclic'

    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0

        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)

            opt.zero_grad()
            with autocast("cuda"):
                output = model(X)
                loss = F.cross_entropy(output, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            if batch_scheduler:
                scheduler.step()

        if not batch_scheduler:
            scheduler.step()

        # Store training mode and evaluate
        was_training = model.training
        model.eval()  # Set to evaluation mode
        test_loss, test_acc = evaluate_standard(test_loader, model)
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 10, 1)
        fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model)
        torch.cuda.empty_cache()
        if was_training:
            model.train()
        try:
            lr = scheduler.get_last_lr()[0]
        except:
            lr = scheduler.get_lr()[0]
        log_metrics(csv_logfile, fieldnames, {
            'epoch': epoch,
            'lr': lr,
            'train_loss': train_loss/train_n,
            'train_acc': train_acc/train_n,
            'test_acc': test_acc,
            'pgd_acc': pgd_acc,
            'fgsm_acc': fgsm_acc
        })
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(
                args.out_dir, f'checkpoint_clean_epoch_{epoch+1}.pth')
            robust_acc_value = robust_acc if 'robust_acc' in locals() else None
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss/train_n,
                'train_acc': train_acc/train_n,
                'test_acc': test_acc,
                'pgd_acc': pgd_acc,
                'fgsm_acc': fgsm_acc,
                'robust_acc': robust_acc_value
            }, checkpoint_path)
            logger.info(
                f'Checkpoint saved at epoch {epoch+1} to {checkpoint_path}')
        if args.early_stop:
            model.eval()
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon,
                                   pgd_alpha, 5, 1, opt, scaler)
            with torch.no_grad():
                output = model(
                    clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                logger.info(
                    f'Early stopping at epoch {epoch}, robust accuracy decreased by more than 0.2')
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            model.train()
        epoch_time = time.time()
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        print('%d \t %.1f \t \t %.4f \t %.4f \t %.4f' % (
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n))
    train_time = time.time()
    if not args.early_stop:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, os.path.join(
        args.out_dir, f'train_clean_output.pth'))
    logger.info('Total train time: %.4f minutes',
                (train_time - start_train_time)/60)
    print('Total train time: %.4f minutes' %
          ((train_time - start_train_time)/60))
    model_test = PreActResNet18().cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    fgsm_loss, fgsm_acc = evaluate_fgsm(test_loader, model_test)
    logger.info(
        'Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t FGSM Loss \t FGSM Acc')
    print('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t FGSM Loss \t FGSM Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                test_loss, test_acc, pgd_loss, pgd_acc, fgsm_loss, fgsm_acc)
    print('%.4f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f' %
          (test_loss, test_acc, pgd_loss, pgd_acc, fgsm_loss, fgsm_acc))
    plot_metrics(csv_logfile, f'clean', args.out_dir)


if __name__ == "__main__":
    main()
