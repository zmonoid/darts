import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from torch.distributions import Categorical


class MixedOp(nn.Module):

    def __init__(self, C, stride, binarize=True):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.binarize = binarize
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        dist = Categorical(logits=weights)
        if self.binarize:
            idx1 = dist.sample().item()
            idx2 = dist.sample().item()
            w = torch.cat((weights[idx1].view(1), weights[idx2].view(1)), dim=0)
            w = F.softmax(w, dim=0)
            return self._ops[idx1](x) * w[0] + self._ops[idx2](x) * w[1]
        else:
            return sum(w * op(x) for w, op in zip(dist.probs, self._ops))


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, binarize=True):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, binarize=binarize)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, binarize=True):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._binarize = binarize

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, binarize=binarize)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def freeze_toggle(self, freeze=True):
        for param in self.stem.parameters():
            param.requires_grad = not freeze

        for param in self.cells.parameters():
            param.requires_grad = not freeze

        for param in self.tnet.parameters():
            param.requires_grad = not freeze

        for param in self.classifier.parameters():
            param.requires_grad = not freeze

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                # weights = F.softmax(self.alphas_reduce, dim=-1)
                weights = self.alphas_reduce
            else:
                weights = self.alphas_normal
                # weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alpha = torch.randn(k * 2, num_ops)
        self.alpha_n = k * num_ops * 2
        self.theta = nn.Parameter(1e-3 * torch.randn(self.alpha_n))

        self.tnet = nn.Sequential(
            nn.Linear(self.alpha_n, self.alpha_n // 4, bias=False),
            # nn.BatchNorm1d(70),
            nn.ReLU(inplace=True),

            nn.Linear(self.alpha_n // 4, self.alpha_n // 4, bias=False),
            # nn.BatchNorm1d(70),
            nn.ReLU(inplace=True),

            nn.Linear(self.alpha_n // 4, self.alpha_n),
        )

        self.step_theta()

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def step_theta(self, theta=None):
        if theta is not None:
            self.theta.data = theta.to(self.theta.device)

        self.alpha = self.tnet(self.theta.view(1, -1)).view_as(self.alpha)
        alphas = torch.split(self.alpha, self.alpha_n // len(PRIMITIVES) // 2, dim=0)
        self.alphas_normal.data = alphas[0]
        self.alphas_reduce.data = alphas[1]

    def grad_theta(self, loss):
        alpha_grad = torch.autograd.grad(loss, (self.alphas_normal, self.alphas_reduce))
        alpha_grad = torch.cat(alpha_grad, dim=0)
        # print(alpha_grad.norm())
        self.alpha.backward(gradient=alpha_grad)
        return self.theta.grad

    def tnet_backward(self):
        alpha_grad = torch.cat((self.alphas_normal.grad, self.alphas_reduce.grad), dim=0)
        self.alpha.backward(gradient=alpha_grad)
        # print(self.theta.grad.norm())
        return self.theta.grad

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

