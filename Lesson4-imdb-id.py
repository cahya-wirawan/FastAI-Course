import pandas as pd
import spacy
import fastai
from fastai.basic_data import *
from fastai.datasets import *
from fastai.text.data import *
from fastai.text.learner import *
from fastai.text import *

import ulmfit
from ulmfit import *
from ulmfit.pretrain_lm import *
from ulmfit.__main__ import *

print(fastai.__version__)
print(spacy.__version__)

LANG = 'id'
path = Path('/mnt/mldata/data/LM/ulmfit/wiki/id-2')
MODEL_NAME = "vf30k"
path.ls()


model_path = Path('/mnt/mldata/data/LM/ulmfit/wiki/id-2/models/vf30k/lstm_orig.m')
#model_path = Path('/mnt/mldata/data/LM/ulmfit/tmp/models/vf30k/lstm_orig.m')
dataset_path = Path('/mnt/mldata/data/LM/ulmfit/tmp/id-2')

#learn = load_learner('/mnt/mldata/data/LM/ulmfit/tmp/id-2/models/vf30k/lstm_orig.m', file='cls_best.pth')
#print(learn)

uf = ULMFiT()
#uf_cls = uf.cls(dataset_path=dataset_path, base_lm_path=model_path,
#                lang=f'{LANG}', name='orig')

l_cls = uf.load_cls(dataset_path/'models/vf30k/lstm_orig.m')
# data_cls, _, _ = l_cls.load_cls_data(40)
data_cls = DataBunch.load_empty('/mnt/mldata/data/LM/ulmfit/tmp/id-2/models/vf30k', 'empty.pkl')
learn = l_cls.create_cls_learner(data_cls)
print("uf:", uf)
#print("uf_cls:", uf_cls)
print("learn:", learn)

"""
data_lm = DataBunch.load_empty(path/f'models/{MODEL_NAME}')
print(data_lm.vocab.itos[:20])
data_lm.path = path

bppt_path = Path("/mnt/mldata/data/LM/id/dataset/BPPTIndToEngCorpus")
data_clas = TextList.from_csv(bppt_path, csv_name="bppt_panl_test.csv", vocab=data_lm.vocab)
data_clas = data_clas.split_by_rand_pct(valid_pct=0.2, seed=10)
# data_clas.inner_df.iloc[:10]

print("Begin")
data_clas = data_clas.label_from_df(cols='target1')
print("End: ", data_clas)
print("data_clas: ", data_clas.train[0])
"""