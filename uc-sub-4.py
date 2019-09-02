import os
import numpy as np
from PIL import Image
from functools import partial
from collections import OrderedDict
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from pathlib import Path

# Load your model to this variable
model = None

# If you used something other than 224x224 cropped images, set the correct size here
image_size = 224
# Values you used for normalizing the images. Default here are for
# pretrained models from torchvision.
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
}

"""
Sequential(
  (0): AdaptiveAvgPool2d(output_size=1)
  (1): AdaptiveMaxPool2d(output_size=1)
  (2): Lambda()
  (3): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (4): Dropout(p=0.25)
  (5): Linear(in_features=4096, out_features=512, bias=True)
  (6): ReLU(inplace)
  (7): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (8): Dropout(p=0.5)
  (9): Linear(in_features=512, out_features=102, bias=True)
)
"""

class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

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
    if (not isinstance(m, bn_types)) and requires_grad(m):
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

# Define model
def create_flower_model():
    model = models.resnet152(pretrained=True)
    print(model)
    for param in model.parameters():
        param.requires_grad = False

    avgpool = nn.Sequential(OrderedDict([
        ('aconpool2d', AdaptiveConcatPool2d()),
        ('lambda', Lambda(lambda x: x.view((x.size(0), -1)))),
    ]))

    model.avgpool = avgpool

    # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model.fc.in_features
    classifier = nn.Sequential(OrderedDict([
        ('bnormal1', nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ('dout1', nn.Dropout(p=0.25)),
        ('lin1', nn.Linear(in_features=4096, out_features=512, bias=True)),
        ('relu', nn.ReLU(inplace=True)),
        ('bnormal2', nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ('dout2', nn.Dropout(p=0.5)),
        ('lin2', nn.Linear(in_features=512, out_features=102, bias=True))
    ]))
    model.fc = classifier
    apply_init(model.avgpool, nn.init.kaiming_normal_)
    apply_init(model.fc, nn.init.kaiming_normal_)
    return model

def load_checkpoint(path_checkpoint):
    checkpoint = torch.load(path_checkpoint, map_location='cpu')
    model = create_flower_model()
    print("model:", model)
    model.class_to_idx = checkpoint['class_to_idx']
    model.class_names = checkpoint['class_names']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model

model = load_checkpoint('resnet152-256-01.pth')

model = model.to(device)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    return data_transforms['valid'](image)

path_data = Path('/mnt/mldata/data/Udacity/flower_data')
path_img_1 = path_data/'test/1/image_06743.jpg'
path_img_2 = path_data/'test/10/image_07104.jpg'
path_img_3 = path_data/'train/30/image_03543.jpg'
batch_size = 10

image_dataset = datasets.ImageFolder(os.path.join(path_data, 'valid'), data_transforms['valid'])
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def predict(path_img, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(path_img)
    image = process_image(image)
    image = image.view(1, 3, image_size, image_size)
    image = image.to(device)
    outputs = model(image)
    prediction = nn.Softmax(dim=0)(outputs[0])
    probability , classes = prediction.topk(topk)
    classes = [model.class_names[i] for i in classes]
    return probability, classes


print(predict(path_img_3, model))
