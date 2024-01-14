# -*- coding: utf-8 -*-
'''
@Time : 2023/7/19 10:58
@Author : KNLiu, MQHuang
@FileName : main.py
@Software : Pycharm
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config
import anndata as ad
import pandas as pd
import scanpy as sc
from utils import seed_everything, save_checkpoints, load_checkpoints
from utils import get_mean, get_hardproportion
from typing import Literal, Optional


# train
def train_MLP(model: nn.Module, train_loader: DataLoader, load):
    # seed everything
    seed_everything(42)

    # initialize model and optimizer
    model = model.to(device=config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.CLASSIFY_LEARNING_RATE,
                           betas=(0.1, 0.999))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=0.001,
                                              steps_per_epoch=len(train_loader),
                                              epochs=config.CLASSIFY_EPOCHS)
    # loss
    classify_criterion = nn.CrossEntropyLoss()

    # weight initialization
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    # start training process
    if load:
        checkpoints = load_checkpoints(config.CLASSIFIER_CHECKPOINT)
        model.load_state_dict(checkpoints["model"])
        optimizer.load_state_dict(checkpoints["optimizer"])

    for epoch in range(config.CLASSIFY_EPOCHS):
        loop = tqdm(enumerate(train_loader), leave=False, total=len(train_loader))
        losses = []
        batch_num = 0
        for idx, (x, label) in loop:
            x = x.to(device=config.DEVICE)  # Get the data to cuda is possible
            label = label.to(device=config.DEVICE)

            # reshape
            x = x.reshape(x.shape[0], -1)
            label = label.reshape(label.shape[0], -1)

            # forward
            x_hat, _ = model(x)
            loss = classify_criterion(x_hat, label)
            losses.append(loss.item())
            batch_num += 1

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(
                f"[Epoch {epoch + 1}/{config.CLASSIFY_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # mean_loss = sum(losses) / batch_num
            scheduler.step()

        if config.SAVE_MODEL and (epoch + 1) % config.CLASSIFY_EPOCHS == 0:
            save_checkpoints(model, optimizer, pth=config.CLASSIFIER_CHECKPOINT)

    print("==> Finish training !")
    return model
