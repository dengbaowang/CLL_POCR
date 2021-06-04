import os
import time
import argparse
import torch
from torch import nn
from torch.backends import cudnn
import dataset as dataset
import torchvision
import torch.nn.functional as F
from wideresnet import WideResNet
from preactresnet import PreActResNet18
from lenet import LeNet
import logging


parser = argparse.ArgumentParser(description='Learning from complamentary labels via partial-output consistency regularization')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lam', default=1, type=float)

parser.add_argument('--dataset', type=str, choices=['svhn', 'cifar10', 'cifar100'], default='cifar10')

parser.add_argument('--model', type=str, choices=['lenet', 'preact', 'widenet'], default='preact')

parser.add_argument('--lr', default=0.1, type=float)

parser.add_argument('--data-dir', default='./data/', type=str)

args = parser.parse_args()
best_prec1 = 0
num_classes = 10


logging.basicConfig(format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler('./result_'+ args.dataset + '__'+ args.model + '_lam_' +str(args.lam) + '.log'),
            logging.StreamHandler()
        ])

logging.info(args)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def comp_train(train_loader, model, optimizer, epoch, consistency_criterion):
    """
        Run one train epoch
    """
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.train()
    for i, (x_aug0, x_aug1, x_aug2, y, comp_y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # complementary label
        comp_y = comp_y.float().cuda()
        # original samples with pre-processing
        x_aug0 = x_aug0.cuda()
        y_pred_aug0 = model(x_aug0)
        # augmentation1
        x_aug1 = x_aug1.cuda()
        y_pred_aug1 = model(x_aug1)
        # augmentation2
        x_aug2 = x_aug2.cuda()
        y_pred_aug2 = model(x_aug2)
        
        y_pred_aug0_probas_log = torch.log_softmax(y_pred_aug0, dim=-1)
        y_pred_aug1_probas_log = torch.log_softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas_log = torch.log_softmax(y_pred_aug2, dim=-1)

        y_pred_aug0_probas = torch.softmax(y_pred_aug0, dim=-1)
        y_pred_aug1_probas = torch.softmax(y_pred_aug1, dim=-1)
        y_pred_aug2_probas = torch.softmax(y_pred_aug2, dim=-1)

        # re-normalization
        revisedY0 = (1 - comp_y).clone()
        revisedY0 = revisedY0 * y_pred_aug0_probas
        revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(num_classes,1).transpose(0,1)
        soft_positive_label0 = revisedY0.detach()
        consist_loss1 = consistency_criterion(y_pred_aug1_probas_log, soft_positive_label0)
        consist_loss2 = consistency_criterion(y_pred_aug2_probas_log, soft_positive_label0)   #Consistency loss

        # complementary loss: SCL-LOG
        comp_loss = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(y_pred_aug0, dim=1)) * comp_y, dim=1))

        # dynamic weighting factor
        lam = min((epoch/100)*args.lam, args.lam)

        # Unified loss
        final_loss = comp_loss + lam * (consist_loss1 + consist_loss2)
        
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        losses.update(final_loss.item(), x_aug0.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lam ({lam})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,lam=lam))

    return losses.avg

def validate(valid_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                logging.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(valid_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


def partial_output_cosistency_training():
    global args, best_prec1

    # load data
    if args.dataset == "cifar10":
        train_loader, test = dataset.cifar10_dataloaders(args.data_dir)
    elif args.dataset == 'svhn':
        train_loader, test = dataset.svhn_dataloaders(args.data_dir)
    
    # load model
    if args.model == 'preact':
        model = PreActResNet18()
    elif args.model == 'widenet':
        model = WideResNet(34, 10, widen_factor=10, dropRate=0.0)
    elif args.model =='lenet':
        model = LeNet(out_dim=10, in_channel=3, img_sz=32)

    model = model.cuda()
    
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch= -1)
    
    cudnn.benchmark = True

    # Train loop
    for epoch in range(0, args.epochs):

        logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        # training
        trainloss = comp_train(train_loader, model, optimizer, epoch, consistency_criterion)
        # lr_step
        scheduler.step()
        # evaluate on validation set
        valacc, valloss = validate(test, model, criterion, epoch)


if __name__ == '__main__':
    partial_output_cosistency_training()


