import time
import os
import copy
import dill
import pickle
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from pathlib import Path
from functools import partial

arch = 'resnet152'
data_dir = Path('/mnt/mldata/data/Udacity/flower_data')
train_dir = data_dir/'train'
valid_dir = data_dir/'valid'

image_size = 224
batch_size = 32

resnet = getattr(models, arch)
model_checkpoint = "{}-{}".format(arch, image_size)
print(model_checkpoint)

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

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
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

def train_model_1cycle(model, criterion, optimizer, dataloaders,
                       lr_min=1e-4, lr_max=1e-3, cycle_length=0.8, num_epochs=25):
    since = time.time()

    iteration = len(dataloaders['train'])*num_epochs
    iteration_cycle = iteration*cycle_length
    iteration_cycle_half = iteration_cycle/2
    lr_step = (lr_max-lr_min)/iteration_cycle_half
    lr_last = lr_min/100.0
    lr_step_last = (lr_min-lr_last)/(iteration-iteration_cycle)
    iteration_counter = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    lr = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("lr:", iteration_counter, lr)
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_i, (inputs, labels) in enumerate(dataloaders[phase]):
                if phase == 'train':
                    #print("iteration_counter:", iteration_counter)
                    if iteration_counter<iteration_cycle_half:
                        lr = lr_min + iteration_counter*lr_step
                    elif iteration_counter<iteration_cycle:
                        lr = lr_max - (iteration_counter-iteration_cycle_half)*lr_step
                    else:
                        lr = lr_min - (iteration_counter-iteration_cycle)*lr_step_last
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    #print("lr:", iteration_counter, lr)
                    iteration_counter += 1

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
    print("lr:", iteration_counter, lr)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def create_flower_model(architectur):
    model = architectur(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_ftrs, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        #('fc2', nn.Linear(4096, 1024)),
        #('relu2', nn.ReLU()),
        #('dropout2', nn.Dropout(p=0.4)),
        ('fc3', nn.Linear(4096, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.fc = classifier
    return model

def load_checkpoint(model_name):
    checkpoint = torch.load(model_name)
    model = create_flower_model()
    model.image_size = checkpoint['image_size']
    model.class_to_idx = checkpoint['class_to_idx']
    model.class_names = checkpoint['class_names']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    return model

def unfreeze(model):
    for name, child in model.named_children():
        print("unfreeze", name)
        for param in child.parameters():
            param.requires_grad = True
        unfreeze(child)

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                              shuffle=True, num_workers=6)
               for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = create_flower_model(resnet)
model = model.to(device)
model_name = model_checkpoint + '-stage-1.pth'
print(model_name)

#criterion = torch.nn.CrossEntropyLoss()
criterion = nn.NLLLoss()
optimizer_ft = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model = train_model_1cycle(model, criterion, optimizer_ft, dataloaders,
                           lr_min=1e-3, lr_max=1e-2, num_epochs=20)

model.class_to_idx = image_datasets['train'].class_to_idx
model.class_names = class_names

checkpoint_dict = {
    'arch': arch,
    'image_size': image_size,
    'class_to_idx': model.class_to_idx,
    'class_names': model.class_names,
    'state_dict': model.state_dict()
}

torch.save(checkpoint_dict, model_name)

criterion = nn.NLLLoss()
optimizer_ft = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
unfreeze(model)
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
                    num_epochs=25)
model_name = model_checkpoint + '-stage-2.pth'
class_to_idx = {v:k for k,v in enumerate(class_names)}
checkpoint_dict = {
    'arch': arch,
    'class_to_idx': class_to_idx,
    'class_names': class_names,
    'state_dict': model.state_dict()
}

torch.save(checkpoint_dict, model_name)