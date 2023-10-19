# The following code is adapted from [simsiam] by [facebookresearch], available at [https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py].
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch.distributed as dist
from loader import JetCollater, JetDataset
from builder import JetCLS, JetCLSMinimal
from sklearn.metrics import roc_auc_score


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.4f')
    acc = AverageMeter('Acc', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (part_energy, edge_head, edge_tail, edge_value, part_indicator, target) in enumerate(
            train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            part_energy = part_energy.cuda(args.gpu, non_blocking=True)
            edge_head = edge_head.cuda(args.gpu, non_blocking=True)
            edge_tail = edge_tail.cuda(args.gpu, non_blocking=True)
            edge_value = edge_value.cuda(args.gpu, non_blocking=True)
            part_indicator = part_indicator.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model([part_energy, edge_head, edge_tail, edge_value, part_indicator])
        loss = criterion(output, target)

        # measure accuracy and record loss
        _acc = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), target.size(0))
        acc.update(_acc[0].item(), target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, args):
    # switch to evaluate mode
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, (part_energy, edge_head, edge_tail, edge_value, part_indicator, target) in enumerate(
                val_loader):
            if args.gpu is not None:
                part_energy = part_energy.cuda(args.gpu, non_blocking=True)
                edge_head = edge_head.cuda(args.gpu, non_blocking=True)
                edge_tail = edge_tail.cuda(args.gpu, non_blocking=True)
                edge_value = edge_value.cuda(args.gpu, non_blocking=True)
                part_indicator = part_indicator.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model([part_energy, edge_head, edge_tail, edge_value, part_indicator])
            y_true.append(target.cpu())
            y_pred.append(torch.nn.functional.softmax(output, dim=-1).cpu())

        y_true = torch.concat(y_true, dim=0)
        y_pred = torch.concat(y_pred, dim=0)
        _auc = roc_auc_score(y_true, y_pred[:, 1], average="macro", multi_class="ovr")
        print("AUC:%.4f" % _auc)

    return _auc



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# model = JetCLS(n_terms=8, n_point=4)
# traindir = os.path.join(args.data, 'train/')
# valdir = os.path.join(args.data, 'val/')
# train_dataset_list = os.listdir(traindir)
# valid_dataset_list = os.listdir(valdir)
# train_dataset = torch.utils.data.ConcatDataset([JetDataset(dataset_file=traindir + x)
#                                                 for x in train_dataset_list])
# valid_dataset = torch.utils.data.ConcatDataset([JetDataset(dataset_file=valdir + x)
#                                                 for x in valid_dataset_list])
# if args.distributed:
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
# else:
#     train_sampler = None
#
# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     num_workers=args.workers,
#     pin_memory=True,
#     collate_fn=JetCollater(r_max=args.radius_max)
# )
# val_loader = torch.utils.data.DataLoader(
#     valid_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=args.workers,
#     pin_memory=True,
#     collate_fn=JetCollater(r_max=args.radius_max)
# )
