from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import num_features_model
import torch
from torch.autograd import Variable
import dill
import pickle
from torchvision import datasets, models, transforms
from PIL import Image

#import onnx

def _resnet_split_2(m:nn.Module): return (m[0][6],m[1])


@dataclass
class Learner_2():
    "Train `model` using `data` to minimize `loss_func` with optimizer `opt_func`."
    data:DataBunch
    model:nn.Module
    train_bn:bool=True
    layer_groups:Collection[nn.Module]=None
    def __post_init__(self)->None:
        "Setup path,metrics, callbacks and ensure model directory exists."
        self.model = self.model.to(torch.device('cpu'))
        if not self.layer_groups: self.layer_groups = [nn.Sequential(*flatten_model(self.model))]

    def split(self, split_on:SplitFuncOrIdxList)->None:
        "Split the model at `split_on`."
        if isinstance(split_on,Callable): split_on = split_on(self.model)
        self.layer_groups = split_model(self.model, split_on)

    def freeze_to(self, n:int)->None:
        "Freeze layers up to layer `n`."
        for g in self.layer_groups[:n]:
            for l in g:
                if not self.train_bn or not isinstance(l, bn_types): requires_grad(l, False)
        for g in self.layer_groups[n:]: requires_grad(g, True)

    def freeze(self)->None:
        "Freeze up to last layer."
        assert(len(self.layer_groups)>1)
        self.freeze_to(-1)

    def load(self, name:PathOrStr, device:torch.device=torch.device('cpu')):
        "Load model `name` from `self.model_dir` using `device`, defaulting to `self.data.device`."
        self.model.load_state_dict(torch.load(name, map_location=device))
        return self

    def get_preds(model:nn.Module, dl:DataLoader, pbar:Optional[PBar]=None, cb_handler:Optional[CallbackHandler]=None,
                  activ:nn.Module=None, loss_func:OptLossFunc=None, n_batch:Optional[int]=None) -> List[Tensor]:
        "Tuple of predictions and targets, and optional losses (if `loss_func`) using `dl`, max batches `n_batch`."
        res = [torch.cat(o).cpu() for o in
               zip(*validate(model, dl, cb_handler=cb_handler, pbar=pbar, average=False, n_batch=n_batch))]
        if loss_func is not None: res.append(calc_loss(res[0], res[1], loss_func))
        if activ is not None: res[0] = activ(res[0])
        return res


    def get_preds(self, ds_type:DatasetType=DatasetType.Valid, with_loss:bool=False, n_batch:Optional[int]=None, pbar:Optional[PBar]=None) -> List[Tensor]:
        "Return predictions and targets on the valid, train, or test set, depending on `ds_type`."
        lf = self.loss_func if with_loss else None
        return get_preds(self.model, self.dl(ds_type), cb_handler=CallbackHandler(self.callbacks),
                         activ=_loss_func2activ(self.loss_func), loss_func=lf, n_batch=n_batch, pbar=pbar)

    def pred_batch(self, ds_type:DatasetType=DatasetType.Valid, pbar:Optional[PBar]=None) -> List[Tensor]:
        "Return output of the model on one batch from valid, train, or test set, depending on `ds_type`."
        dl = self.dl(ds_type)
        nw = dl.num_workers
        dl.num_workers = 0
        preds,_ = self.get_preds(ds_type, with_loss=False, n_batch=1, pbar=pbar)
        dl.num_workers = nw
        return preds

class ClassificationLearner_2(Learner_2):
    def predict(self, img:Image):
        "Return prect class, label and probabilities for `img`."
        ds = self.data.valid_ds
        ds.set_item(img)
        res = self.pred_batch()[0]
        ds.clear_item()
        pred_max = res.argmax()
        return self.data.classes[pred_max],pred_max,res

def create_cnn_2(arch:Callable, n_classes:int=2, cut:Union[int,Callable]=None, pretrained:bool=True,
               lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
               custom_head:Optional[nn.Module]=None, split_on:Optional[SplitFuncOrIdxList]=None,
               classification:bool=True, **kwargs:Any)->Learner:
    "Build convnet style learners."
    assert classification, 'Regression CNN not implemented yet, bug us on the forums if you want this!'
    meta = {'cut':-2, 'split':_resnet_split_2 }
    body = create_body(arch(pretrained), ifnone(cut,meta['cut']))
    nf = num_features_model(body) * 2
    head = custom_head or create_head(nf, n_classes, lin_ftrs, ps)
    model = nn.Sequential(body, head)
    #learner_cls = ifnone(data.learner_type(), ClassificationLearner_2)
    learner_cls = ClassificationLearner_2
    learn = learner_cls(data, model, **kwargs)
    learn.split(ifnone(split_on,meta['split']))
    if pretrained: learn.freeze()
    apply_init(model[1], nn.init.kaiming_normal_)
    return learn

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

#data = ImageDataBunch.from_folder(path_data, ds_tfms=get_transforms(), valid='test', size=224, bs=bs)
#data.normalize(imagenet_stats)
#learn = create_cnn_2(models.resnet34, data.c)
#learn.load(path_model)
#img0 = open_image(path_img)
img = Image.open(path_img)
img_tensor = content_transform(img)
img_tensor = Variable(img_tensor).view(-1, 3, input_size, input_size)

images = [content_transform(Image.open(i)) for i in path_images]
images_tensor = Variable(torch.cat(images))
images_tensor = images_tensor.view(-1, 3, input_size, input_size)


path_model_resnet = Path("resnet34.pth")
#torch.save(learn.model, path_model_resnet, pickle_module=dill)
model = torch.load(path_model_resnet, pickle_module=dill)
model.eval()

#print(model)
outputs = model(img_tensor)
id = torch.argmax(outputs[0])
print(id, img_classes[id])
outputs = model(images_tensor)
for o in outputs:
    id = torch.argmax(o)
    print(id, img_classes[id])
print("end")

#preds = learn.predict(img)
#print(preds)

#img = learn.data.train_ds[300][0]
#preds = learn.predict(img)

#log_preds_valid, y_valid = learn.get_preds(ds_type=DatasetType.Valid)
#print(accuracy(log_preds_valid, y_valid))

