
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import numpy as np
import pandas as pd

from tqdm import tqdm
import neptune
import random
import os
import glob
import torch
import utils
import logging



from model_meta import MetaLearner
from utils import timestr

args = utils.read_yaml().parameters




def main():

    seed(args.seed)

    now = timestr()
    neptune.set_property('Time Str', now)

    mini_train_queque, mini_test_queque, cifar_train_queque, cifar_test_queque = parepare_dataset()

    learner = MetaLearner(args)
    learner.cuda()
    params = [utils.count_parameters_in_MB(m) for m in [learner.inet, learner.cnet]]
    print(f"Params: INet {params[0]}, CNet {params[1]}")

    for epoch in range(args.epochs):

        learner.scheduler_i.step()
        learner.scheduler_c.step()
        lr = learner.scheduler_i.get_lr()[0]

        print('epoch %d lr %e', epoch, lr)

        genotype = learner.cnet.module.genotype()
        print('CIFAR genotype = %s', genotype)

        genotype = learner.inet.module.genotype()
        print('MINI genotype = %s', genotype)


        print(F.softmax(learner.cnet.module.alphas_normal, dim=-1))
        print(F.softmax(learner.cnet.module.alphas_reduce, dim=-1))
        print(F.softmax(learner.inet.module.alphas_normal, dim=-1))
        print(F.softmax(learner.inet.module.alphas_reduce, dim=-1))
        print(learner.theta)

        # training
        train(cifar_train_queque, mini_train_queque, learner)

        # validation
        infer(cifar_test_queque, mini_test_queque, learner)

        torch.save(
            {
                'cnet': learner.cnet.module.state_dict(),
                'inet': learner.inet.module.state_dict(),
                'alpha': learner.theta
            },
            f'ckpt/model_{now}_{epoch}.pth'
        )


def train(cifar_train_queque, mini_train_queque, learner):


    learner.cnet.train()
    learner.inet.train()

    c_itr = iter(cifar_train_queque)
    i_itr = iter(mini_train_queque)

    steps = max(len(mini_train_queque), len(cifar_train_queque))

    count_c = 0
    losses_c = 0
    correct_c = 0

    count_i = 0
    losses_i = 0
    correct_i = 0

    for step in tqdm(range(steps)):
        try:
            xc, yc = next(c_itr)
        except StopIteration:
            c_itr = iter(cifar_train_queque)
            xc, yc = next(c_itr)

        try:
            xi, yi = next(i_itr)
        except StopIteration:
            i_itr = iter(mini_train_queque)
            xi, yi = next(i_itr)


        loss_i, loss_c, logits_i, logits_c = learner.meta_step(
            xi.cuda(non_blocking=True), yi.cuda(non_blocking=True),
            xc.cuda(1, non_blocking=True), yc.cuda(1, non_blocking=True), args.meta_lr)


        prec1, prec5 = utils.accuracy(logits_i.cpu(), yi, topk=(1, 5))
        neptune.send_metric('mini batch end loss', loss_i.item())
        neptune.send_metric('mini batch end acc@1', prec1.item())
        neptune.send_metric('mini batch end acc@5', prec5.item())


        prec1, prec5 = utils.accuracy(logits_c.cpu(), yc, topk=(1, 5))
        neptune.send_metric('cifar batch end loss', loss_c.item())
        neptune.send_metric('cifar batch end acc@1', prec1.item())
        neptune.send_metric('cifar batch end acc@5', prec5.item())

        losses_c += loss_c.item() * xc.size(0)
        count_c += xc.size(0)
        correct_c += logits_c.argmax(dim=-1).cpu().eq(yc).sum().item()

        losses_i += loss_i.item() * xi.size(0)
        count_i += xi.size(0)
        correct_i += logits_i.argmax(dim=-1).cpu().eq(yi).sum().item()


    epoch_loss_c = losses_c / count_c
    epoch_acc_c = correct_c * 1.0 / count_c
    neptune.send_metric('cifar epoch end train loss', epoch_loss_c)
    neptune.send_metric('cifar epoch end train acc', epoch_acc_c)

    epoch_loss_i = losses_i / count_i
    epoch_acc_i = correct_i * 1.0 / count_i
    neptune.send_metric('mini epoch end train loss', epoch_loss_i)
    neptune.send_metric('mini epoch end train acc', epoch_acc_i)

    print(f"Training: MINI Acc {epoch_acc_i}, CIFAR Acc {epoch_acc_c}, "
          f"MINI Loss {epoch_loss_i}, CIFAR Loss {epoch_loss_c}")



def infer(cifar_test_queque, mini_test_queque, learner):


    learner.cnet.eval()
    learner.inet.eval()

    c_itr = iter(cifar_test_queque)
    i_itr = iter(mini_test_queque)

    steps = max(len(cifar_test_queque), len(mini_test_queque))

    count_c = 0
    correct_c = 0

    count_i = 0
    correct_i = 0

    with torch.no_grad():
        for step in tqdm(range(steps)):

            try:
                xc, yc = next(c_itr)
            except StopIteration:
                c_itr = iter(cifar_test_queque)
                xc, yc = next(c_itr)

            try:
                xi, yi = next(i_itr)
            except StopIteration:
                i_itr = iter(mini_test_queque)
                xi, yi = next(i_itr)

            if xi is not None:
                logits_i = learner.inet(xi.cuda(non_blocking=True))
                count_i += xi.size(0)
                correct_i += logits_i.argmax(dim=-1).cpu().eq(yi).sum().item()

            if xc is not None:
                logits_c = learner.cnet(xc.cuda(non_blocking=True))
                count_c += xc.size(0)
                correct_c += logits_c.argmax(dim=-1).cpu().eq(yc).sum().item()


    epoch_acc_c = correct_c * 1.0 / count_c
    epoch_acc_i = correct_i * 1.0 / count_i

    neptune.send_metric('cifar epoch validation acc', epoch_acc_c)
    neptune.send_metric('mini epoch validation acc', epoch_acc_i)

    print(f"Validation: MINI Acc {epoch_acc_i}, CIFAR Acc {epoch_acc_c}")

def parepare_dataset():

    train_transform, valid_transform = utils._data_transforms_cifar100(args)

    cifar_train_data = dset.CIFAR100(
        root=args.data,
        train=True,
        download=True,
        transform=train_transform
    )

    train_transform, valid_transform = utils._data_transforms_mini(args)
    mini_train_data = dset.ImageFolder(
        root=os.path.join(args.data, 'miniimagenet', 'train_'),
        transform=train_transform
    )

    num_train = len(mini_train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    mini_train_queque = torch.utils.data.DataLoader(
        mini_train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    mini_test_queque = torch.utils.data.DataLoader(
        mini_train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=2)

    cifar_train_queque = torch.utils.data.DataLoader(
        cifar_train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    cifar_test_queque = torch.utils.data.DataLoader(
        cifar_train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=2)

    return mini_train_queque, mini_test_queque, cifar_train_queque, cifar_test_queque

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    np.random.seed(args.seed)

if __name__ == '__main__':
    neptune.init(project_qualified_name='zhoubinxyz/das')
    upload_files = glob.glob('*.py') + glob.glob('*.yaml')
    neptune.create_experiment(params=args, upload_source_files=upload_files)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main()


