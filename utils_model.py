from functools import partial
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn


utils_bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def children(m:nn.Module):
    "Get children of `m`."
    return list(m.children())


def requires_grad(m:nn.Module, b=None):
    "If `b` is not set `requires_grad` on all params in `m`, else return `requires_grad` of first param."
    ps = list(m.parameters())
    if not ps: return None
    if b is None: return ps[0].requires_grad
    for p in ps: p.requires_grad=b


def cond_init(m:nn.Module, init_func):
    "Initialize the non-batchnorm layers of `m` with `init_func`"
    if (not isinstance(m, utils_bn_types)) and requires_grad(m):
        if hasattr(m, 'weight'): init_func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)


def apply_leaf(m:nn.Module, f):
    "Apply `f` to children of `m`."
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    for l in c: apply_leaf(l,f)


def apply_init(m, init_func):
    "Initialize all non-batchnorm layers of `m` with `init_func`."
    apply_leaf(m, partial(cond_init, init_func=init_func))


def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0., actn=None):
    "`n_in`->bn->dropout->linear(`n_in`,`n_out`)->`actn`"
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)


def Flatten():
    "Flattens `x` to a single dimension, often used at the end of a model."
    return Lambda(lambda x: x.view((x.size(0), -1)))


def create_body(model:nn.Module, cut=None, body_fn=None):
    "Cut off the body of a typically pretrained `model` at `cut` or as specified by `body_fn`."
    return (nn.Sequential(*list(model.children())[:cut]) if cut
            else body_fn(model) if body_fn else model)


def create_head(nf:int, nc:int, ps:float=0.5):
    """Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes.
    :param ps: dropout, can be a single float or a list for each layer."""
    lin_ftrs = [nf, 512, nc]
    ps = [ps]
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,True,p,actn)
    return nn.Sequential(*layers)


def create_cnn():
    resnet = models.resnet18(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    num_ftrs = resnet.fc.in_features
    body = create_body(resnet, -2)
    head = create_head(num_ftrs * 2, 102)
    model = nn.Sequential(body, head)
    apply_init(model[1], nn.init.kaiming_normal_)
    print(len(model))
    return model