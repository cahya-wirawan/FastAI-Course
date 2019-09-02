from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import num_features_model
import torch
from torch.autograd import Variable
import dill
import pickle
from torchvision import datasets, models, transforms
from PIL import Image


input_size = 224

bs = 128
path_data = Path('/mnt/mldata/data/Udacity/flower_data')
path_model = path_data/'models/stage-1'

data = ImageDataBunch.from_folder(path_data, ds_tfms=get_transforms(), valid='test', size=224, bs=bs)
data.normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34)
learn.load(path_model)

path_model_resnet = Path("resnet34-1.pth")
torch.save(learn.model, path_model_resnet, pickle_module=dill)
model = torch.load(path_model_resnet, pickle_module=dill)

print("end")
