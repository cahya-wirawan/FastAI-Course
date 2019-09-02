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



bs = 10
path_data = Path('/mnt/mldata/data/Udacity/flower_data')
path_model = path_data/'models/stage-2.pth'
path_img = path_data/'test/1/image_06743.jpg'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ds_tfms=get_transforms()

data = ImageDataBunch.from_folder(path_data, ds_tfms=ds_tfms, valid='valid', size=500, bs=bs)
data.normalize(imagenet_stats)
print(data.train_ds.classes)
learn = create_cnn(data, models.resnet152, metrics=error_rate)
learn.load("stage-2-152")
img = Image.open(path_img)
img_tensor = content_transform(img)
img_tensor = Variable(img_tensor).view(1, 3, 224, 224)

log_preds_valid, y_valid = learn.get_preds(ds_type=DatasetType.Valid)
print(accuracy(log_preds_valid, y_valid))

def predict_images(model, dataloader, classes):
    running_corrects = 0
    for inputs, labels in dataloader:
        #print("labels 0:", labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = np.array([classes[i] for i in labels.data])
        #print("labels 1:", labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        #print("preds:", preds)
        prediction = np.array([model.class_names[i] for i in preds])
        #print("prediction:", prediction)
        running_corrects += np.sum(prediction == labels)

    accuracy = running_corrects / len(dataloader.dataset)
    return accuracy

#acc = predict_images(learn.model, learn.data.valid_dl, learn.data.classes)