# README

**Mingxi Chen, Juan Li** 

## Pre-trained part

The code used for pre-trained model is in the folder `pretrain`.

 `model.py` is the model file and `train.py` is used to run and save models.

 `netD_490.pth` is the saved weights file of discriminative model.

 `netG_490.pth` is the saved weights file of generative model.

## Downstream task part

The code used for the training of downstream task is in the folder `downstream`.

`models/crnn_pre.py `, `train_pre.py`  are uesd to train our model with pre-trained.

`models/crnn.py`,  `train_crnn.py` are used to train the pure CRNN model without pre-trained.

`pre_crnn.pth` is the saved weights file of model with pre-trained.

`crnn.pth` is the saved weights file of model without pre-trained. 