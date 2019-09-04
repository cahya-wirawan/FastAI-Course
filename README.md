# FastAI-Course
Collection of notes and notebooks for FastAI Course V3.
## ULMFiT
It is [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146). 
I created several ulmfit-* notebooks to make it easier to create a language model and its classifier. 
It uses ULMFiT classes from [ulmfit-multilingual](https://github.com/n-waves/ulmfit-multilingual) 
created by Piotr Czapla. Therefore to run the noteboook, please clone the 
[ulmfit-multilingual](https://github.com/n-waves/ulmfit-multilingual) repository, and make a softlink
from ulmfit directory of ulmfit-multilingual to the current directory:

```
$ git clone git@github.com:cahya-wirawan/FastAI-Course.git
$ git clone https://github.com/n-waves/ulmfit-multilingual.git
$ cd FastAI-Course
$ ln -s ../ulmfit-multilingual/ulmfit
```
### Requirements
- python >= 3.6
- pytorch >= 1.0
- fastai >= 1.0.52
