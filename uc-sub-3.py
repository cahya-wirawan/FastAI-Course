import os
import numpy as np
from PIL import Image
from functools import partial
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from pathlib import Path

# Load your model to this variable
model = None

# If you used something other than 224x224 cropped images, set the correct size here
image_size = 500
# Values you used for normalizing the images. Default here are for
# pretrained models from torchvision.
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("pytorch version:", torch.__version__)
print("device:", device)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'valid': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
}

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
    resnet = models.resnet152(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    num_ftrs = resnet.fc.in_features
    body = create_body(resnet, -2)
    head = create_head(num_ftrs * 2, 102)
    model = nn.Sequential(body, head)
    apply_init(model[1], nn.init.kaiming_normal_)
    #print(len(model))
    return model

def load_checkpoint(path_checkpoint):
    checkpoint = torch.load(path_checkpoint, map_location='cpu')
    #checkpoint = torch.load(path_checkpoint)
    #print(checkpoint)
    model = create_cnn()
    model.class_to_idx = checkpoint['class_to_idx']
    model.class_names = checkpoint['class_names']
    #print("Model Original", model.state_dict().keys())
    #print("--------------------------")
    #print("Model Chekcpoint", checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model

model_path ='/home/cahya/Work/Machine Learning/FastAI/FastAI-Course/resnet152-224-cp.pth'
model = load_checkpoint(model_path)

model = model.to(device)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    return data_transforms['valid'](image)

path_data = Path('/data')
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

    _, preds = torch.max(outputs, 1)
    prediction = [model.class_names[i] for i in preds]
    print("preds", preds)
    print("prediction", prediction)

    prediction = nn.Softmax(dim=0)(outputs[0])
    probability , classes = prediction.topk(topk)
    probability = [i for i in probability]
    #print(probability, ":", classes)
    classes = [model.class_names[i] for i in classes]
    return probability, classes

print(predict(path_img_3, model))

def predict_images(path_images, data_transform, batch_size=16, shuffle=False, num_workers=1):
    image_dataset = datasets.ImageFolder(path_images, data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers)
    running_corrects = 0
    counter = 0
    for inputs, labels in dataloader:
        print(counter)
        counter += 1
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = np.array([image_dataset.classes[i] for i in labels.data])
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        prediction = np.array([model.class_names[i] for i in preds])
        running_corrects += np.sum(prediction == labels)

    accuracy = running_corrects / len(dataloader.dataset)
    return accuracy

accuracy = predict_images(os.path.join(path_data, 'valid'), data_transforms['valid'], batch_size=2)
print("accuracy:", accuracy)