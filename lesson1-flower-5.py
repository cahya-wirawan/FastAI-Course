from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import num_features_model
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

bs = 128
path_data = Path('/mnt/mldata/data/Udacity/flower_data')
path_model = path_data/'models/stage-2.pth'
path_img = path_data/'test/1/image_06743.jpg'

data = ImageDataBunch.from_folder(path_data, ds_tfms=get_transforms(), valid='test', size=224, bs=bs)
data.normalize(imagenet_stats)
print(data.train_ds.classes)
learn = create_cnn(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(5, max_lr=slice(1e-3))



#learn.load('stage-2')
#img0 = open_image(path_img)
img = Image.open(path_img)
img_tensor = content_transform(img)
#img_tensor = transforms.ToTensor()(img)
img_tensor = Variable(img_tensor).view(1, 3, 224, 224)


path_model_resnet = Path("resnet34.pth")
#torch.save(learn.model, path_model_resnet, pickle_module=dill)
model = torch.load(path_model_resnet, pickle_module=dill)
model.eval()

print(model)
outputs = model(img_tensor)
print(outputs, torch.argmax(outputs[0]))

#preds = learn.predict(img)
#print(preds)

#img = learn.data.train_ds[300][0]
#preds = learn.predict(img)

#log_preds_valid, y_valid = learn.get_preds(ds_type=DatasetType.Valid)
#print(accuracy(log_preds_valid, y_valid))

