# -*- coding: utf-8 -*-
'''
@Time : 2023/7/19 10:58
@Author : KNLiu, MQHuang
@FileName : config.py
@Software : Pycharm
'''
import torch

# TRAIN_EPOCHS = 5
EPOCHS = 5 #propDEC_end2end
# PROP_EPOCHS = 5
# LEARNING_RATE = 1e-3
INPUT_SIZE = 3000 #mlp
SEED=42

DEVICE = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")  
NUM_WORKERS = 9 #classify num
CLASSIFY_EPOCHS = 7 #propDEC_2stage
DEC_EPOCHS = 5 #propDEC_2stage
BATCH_SIZE = 128  #512
DEC_LEARNING_RATE = 1e-4
CLASSIFY_LEARNING_RATE = 1e-3
EMB_SIZE = 128
MOVING_SPEED = 0.5
EPS = 1e-6
DATA_ROOT = "/volume1/home/mhuang/cellTypeAbundance/data"
CACHE_PATH = "/volume1/home/mhuang/cellTypeAbundance/data/cache"
CLASSIFIER_CHECKPOINT = "./tmp/classifier.pth.tar"
DEC_CHECKPOINT = "./tmp/dec.pth.tar"
PROPDEC_CHECKPOINT = "./tmp/propDEC.pth.tar"
LOAD_MODEL = False
SAVE_MODEL = True

