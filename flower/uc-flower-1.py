import time
import os
import copy
from functools import partial
from collections import OrderedDict
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.optim import lr_scheduler
from pathlib import Path
import numpy as np

arch = 'resnet152'
data_dir = '/mnt/mldata/data/Udacity/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
image_size = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

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

def create_flower_model_1():
    model = models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs, 4096)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.fc = classifier
    return model

# Define model
def create_flower_model_2():
    model = models.resnet152(pretrained=True)
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
        ('dout1', nn.Dropout(p=0.1)),
        ('lin1', nn.Linear(in_features=4096, out_features=512, bias=True)),
        ('relu', nn.ReLU(inplace=True)),
        ('bnormal2', nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
        ('dout2', nn.Dropout(p=0.2)),
        ('lin2', nn.Linear(in_features=512, out_features=102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.fc = classifier
    apply_init(model.avgpool, nn.init.kaiming_normal_)
    apply_init(model.fc, nn.init.kaiming_normal_)
    return model

def load_checkpoint(path_checkpoint):
    checkpoint = torch.load(path_checkpoint, map_location='cpu')
    model = create_flower_model_1()
    print("model:", model)
    model.class_to_idx = checkpoint['class_to_idx']
    model.class_names = checkpoint['class_names']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    return data_transforms['valid'](image)

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

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes


model = create_flower_model_1()
print(model)
model = model.to(device)

#criterion = torch.nn.CrossEntropyLoss()
criterion = nn.NLLLoss()

#learning_rate = 1e-3
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=50)

path_data = Path('/mnt/mldata/data/Udacity/flower_data')
path_img_1 = path_data/'test/1/image_06743.jpg'
path_img_2 = path_data/'test/10/image_07104.jpg'
path_img_3 = path_data/'train/30/image_03543.jpg'
batch_size = 10

image_dataset = datasets.ImageFolder(os.path.join(path_data, 'valid'), data_transforms['valid'])
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


model_ft.class_to_idx = image_datasets['train'].class_to_idx
model_ft.class_names = class_names

arch = "resnet152"
model_name = arch + "-256-test-1.pth"
print(arch, model_name)

checkpoint_dict = {
    'arch': arch,
    'class_to_idx': model_ft.class_to_idx,
    'class_names': model_ft.class_names,
    'state_dict': model_ft.state_dict()
}

torch.save(checkpoint_dict, model_name)

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

accuracy = predict_images(os.path.join(path_data, 'valid'), data_transforms['valid'], batch_size=10)

print(accuracy)
