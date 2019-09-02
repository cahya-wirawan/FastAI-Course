# Load your model to this variable
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

print(torch.__version__)
model = models.resnet18(pretrained=True)
for i, param in model.named_parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 102)




checkpoint = torch.load('resnet18.pth', map_location='cpu')
#b = OrderedDict({k.replace('.num_batches_tracked', ''):v for k, v in c.items()})
model.class_to_idx = checkpoint['class_to_idx']
model.class_names = checkpoint['class_names']
model.load_state_dict(checkpoint['state_dict'])
model.eval()

print(model.class_to_idx)
print(model.class_names)
# If you used something other than 224x224 cropped images, set the correct size here
image_size = 224
# Values you used for normalizing the images. Default here are for
# pretrained models from torchvision.
norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]
