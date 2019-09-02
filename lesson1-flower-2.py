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

content_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

img_classes = ['43', '92', '88', '91', '95', '29', '19', '85', '9', '74', '82', '58',
               '41', '51', '93', '53', '14', '87', '5', '23', '76', '70', '42', '97',
               '90', '32', '69', '98', '39', '16', '61', '68', '83', '96', '34', '4',
               '25', '102', '33', '52', '67', '49', '71', '63', '22', '80', '47', '40',
               '11', '64', '37', '36', '86', '17', '66', '65', '59', '45', '10', '78',
               '94', '31', '77', '7', '2', '56', '3', '72', '21', '35', '26', '100',
               '20', '57', '55', '28', '50', '15', '18', '48', '38', '12', '27', '46',
               '79', '13', '6', '60', '54', '1', '8', '101', '73', '81', '62', '89',
               '99', '24', '75', '84', '44', '30']

bs = 128
path_data = Path('/mnt/mldata/data/Udacity/flower_data')
path_model = path_data/'models/stage-2.pth'
path_img = path_data/'train/10/image_07120.jpg'
path_images = [path_data/'train/10/image_07120.jpg',
               path_data/'train/20/image_04939.jpg']

"""
data = ImageDataBunch.from_folder(path_img, ds_tfms=get_transforms(), test='test', size=224, bs=bs)
data.normalize(imagenet_stats)

path_test = ImageFileList.from_folder(path_img/'test').label_from_folder()

print(path_test)
print(data.train_ds)
print(data.classes)
"""


#image_dataset = datasets.ImageFolder('/tmp/test/', content_transform)
image_dataset = datasets.ImageFolder(str(path_data/'test/'), content_transform)
dataloader_dict = torch.utils.data.DataLoader(image_dataset, batch_size=100, shuffle=False, num_workers=4)

path_model_resnet = Path("resnet34-2.pth")
model = torch.load(path_model_resnet, pickle_module=dill)
model.eval()
print(torch.cuda.is_available())
model.cuda()

accuracy = 0
for i_batch, sample_batched in enumerate(dataloader_dict):
    print(i_batch)
    labels = np.array([image_dataset.classes[i] for i in sample_batched[1]])
    print(labels)
    outputs = model(sample_batched[0].cuda())
    predicted = np.array([img_classes[torch.argmax(o)] for o in outputs])
    print(predicted)
    accuracy += (labels == predicted).sum()

total = len(image_dataset.imgs)
accuracy = 100.0*accuracy/total
print(total, accuracy)
print("end")
