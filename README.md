# FastAI-Course
Collection of notes and notebooks for FastAI Course V3.
## ULMFiT
It is [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146). 
I created several ulmfit-* notebooks to make it easier to create a language model and its classifier. 
It uses ULMFiT classes from [ulmfit-multilingual](https://github.com/n-waves/ulmfit-multilingual) 
created by Piotr Czapla et al. We put the ULMFiT classes as submodule within this repository, 
if you clone it for the first time, you should use the option --recursive as follow:
```
$ git clone --recursive https@github.com:cahya-wirawan/FastAI-Course.git
```
But if you have cloned it before, you have to fetch the submodule manually:
```
$ git pull
$ git submodule update --init --recursive
```
### Requirements
- python >= 3.6
- pytorch >= 1.0
- fastai >= 1.0.52
