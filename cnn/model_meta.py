# Created by zhoub at 3/15/2019

# Enter feature description here

# Enter steps here

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_search import Network
import numpy as np
import utils
import time
from genotypes import PRIMITIVES


class MetaLearner:
    def __init__(self, args):
        self.criteria = nn.CrossEntropyLoss()

        self.cnet = Network(args.init_channels, 100, args.layers, self.criteria, binarize=args.binarize)
        self.inet = Network(args.init_channels, 100, args.layers, self.criteria, binarize=args.binarize)

        self.theta = nn.Parameter(torch.randn(self.inet.alpha_n))
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay

        self.optimizer_c = torch.optim.SGD(self.cnet.parameters(),
                                           args.learning_rate,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay)

        self.optimizer_i = torch.optim.SGD(self.inet.parameters(),
                                           args.learning_rate,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay)

        self.theta_optimizer = torch.optim.Adam((self.theta,),
                                                lr=args.arch_learning_rate,
                                                betas=(0.5, 0.999),
                                                weight_decay=args.arch_weight_decay)

        self.scheduler_i = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_i, float(args.epochs), eta_min=args.learning_rate_min)

        self.scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_c, float(args.epochs), eta_min=args.learning_rate_min)

        self.ratio = args.meta_ratio
        self.arch_learning_rate = args.arch_learning_rate
        self.grad_clip = args.grad_clip

    def cuda(self):
        self.cnet = self.cnet.cuda(1)
        self.cnet = nn.DataParallel(self.cnet, device_ids=[1])

        self.inet = nn.DataParallel(self.inet, device_ids=[0, 2, 3, 4]).cuda()
        self.theta = self.theta.cuda()

    def freeze_toggle(self, freeze=True):
        self.inet.module.freeze_toggle(freeze)
        self.cnet.module.freeze_toggle(freeze)

    def loss(self, xi, yi, theta_i, xc, yc, theta_c):
        self.cnet.module.step_theta(theta_c)
        self.inet.module.step_theta(theta_i)
        logits_c = self.cnet(xc)
        logits_i = self.inet(xi)
        loss_c = self.criteria(logits_c, yc)
        loss_i = self.criteria(logits_i, yi)
        return loss_i, loss_c, logits_i, logits_c

    def meta_step(self, xi, yi, xc, yc, lr):
        self.freeze_toggle(freeze=True)

        loss_i, loss_c, _, _ = self.loss(xi, yi, self.theta, xc, yc, self.theta)
        theta_grad_c = self.cnet.module.grad_theta(loss_c).to(self.theta.device)
        theta_grad_i = self.inet.module.grad_theta(loss_i).to(self.theta.device)

        theta_new_c = self.theta - theta_grad_c * lr
        theta_new_i = self.theta - theta_grad_i * lr

        loss_i, loss_c, _, _ = self.loss(xi, yi, theta_new_i, xc, yc, theta_new_c)

        theta_grad_c_tp1 = self.cnet.module.grad_theta(loss_c).to(self.theta.device)
        theta_grad_i_tp1 = self.inet.module.grad_theta(loss_i).to(self.theta.device)

        self.freeze_toggle(freeze=False)

        loss_i, loss_c, logits_i, logits_c = self.loss(xi, yi, self.theta, xc, yc, self.theta)

        self.optimizer_i.zero_grad()
        self.optimizer_c.zero_grad()
        self.theta_optimizer.zero_grad()
        loss_i.backward()
        loss_c.backward()
        self.cnet.module.tnet_backward()
        self.inet.module.tnet_backward()
        self.theta.grad = self.ratio * theta_grad_c_tp1 + (1 - self.ratio) * theta_grad_i_tp1
        self.optimizer_i.step()
        self.optimizer_c.step()
        self.theta_optimizer.step()

        return loss_i, loss_c, logits_i, logits_c


def main():
    pass


if __name__ == '__main__':
    # net = Network(48, 100, 14, nn.CrossEntropyLoss())
    # params = net.arch_parameters()
    # print(params[0].shape, params[1].shape)
    args = utils.read_yaml().parameters


    learner = MetaLearner(args)
    learner.cuda()

    for _ in range(10):
        now = time.time()
        batch_size = 64
        xi, yi = torch.rand(batch_size, 3, 64, 64), torch.randint(0, 100, (batch_size,))
        xc, yc = torch.rand(batch_size, 3, 32, 32), torch.randint(0, 100, (batch_size,))

        loss_i, loss_c, logits_i, logits_c = learner.meta_step(
            xi.cuda(), yi.cuda(), xc.cuda(1), yc.cuda(1), 0.1)

        print(f'{time.time() - now:4f}\t {loss_i.item():4f}\t {loss_c.item():4f}')

