# -*- coding: utf-8 -*-
'''
@Time : 2023/7/19 10:58
@Author : KNLiu, MQHuang
@FileName : model.py
@Software : Pycharm
'''
import torch
import torch.nn as nn
import config
import anndata as ad
from collections import OrderedDict
from utils import get_mean, seed_everything
from typing import Literal, Optional
from sklearn.cluster import KMeans


class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size, last: bool = False):
        super(LinearBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=input_size, out_features=output_size, bias=False),
            nn.BatchNorm1d(output_size),
            nn.Identity() if last else nn.ReLU()
        )

    def forward(self, x):
        return self.model(x.float())


class MLP(nn.Module):
    def __init__(self, num_classes, input_size: int = config.INPUT_SIZE, emb_size: int = 128):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.emb_size = emb_size

        features = [input_size, 1024, 512, 256, emb_size]

        # Encoder
        layers = []
        for i in range(len(features) - 1):
            if i == len(features) - 2:
                layers.append(LinearBlock(features[i], features[i + 1], last=True))
            else:
                layers.append(LinearBlock(features[i], features[i + 1]))
        encoder_layers_dict = [("Block" + str(i), layers[i]) for i in range(len(layers))]
        self.Encoder = nn.Sequential(OrderedDict(encoder_layers_dict))

        self.linear = nn.Linear(in_features=emb_size, out_features=num_classes)

    def forward(self, x):
        emb = self.Encoder(x)
        scores = self.linear(emb)
        return scores, emb

    def extract_emb(self, adata: ad.AnnData):
        data = adata.X.reshape(adata.X.shape[0], -1)
        data = torch.tensor(data).to(device=config.DEVICE)
        emb = self.Encoder(data)

        adata.obsm["emb"] = emb.cpu().detach().numpy()
        return emb


class SoftAssignment(nn.Module):
    '''可以直接用softmax来算概率，也可以像底下一样，先算每个样本和center的距离（t-sne或cosine或取平均），再将距离转成概率'''
    def __init__(self,
                 ref_adata: ad.AnnData,
                 num_classes: int,
                 emb_dimension: int,
                 metric=Literal["t-sne", "cosine"],
                 alpha: float = 1.0,
                 cluster_centroid: Optional[torch.Tensor] = None):
        super(SoftAssignment, self).__init__()
        self.num_classes = num_classes
        self.emb_dimension = emb_dimension
        self.metric = metric
        self.alpha = alpha

        seed_everything(42)

        assert self.metric == "t-sne" or self.metric == "cosine"
        if cluster_centroid is None:
            if ref_adata.obsm["emb"] is not None:
                self.cluster_centroid = get_mean(ref_adata)
            else:
                raise RuntimeError("expect to extract embedding first")
        else:
            self.cluster_centroid = cluster_centroid

    def forward(self, test_emb) -> torch.Tensor:
        '''
        Compute the soft assignment for a batch of embedding vectors, return a batch of
        soft assignment for each sample in test dataset
        :param test_emb
        :return FloatTensor [batch_size, number of classes]
        '''
        if self.metric == "t-sne":
            norm_squared = torch.sum((test_emb.unsqueeze(1) - self.cluster_centroid) ** 2, 2)
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
            power = float(self.alpha + 1) / 2
            numerator = numerator ** power
            return numerator / torch.sum(numerator, dim=1, keepdim=True)
        elif self.metric == "cosine":
            scale = torch.sqrt(torch.FloatTensor([self.emb_dimension])).to(device=config.DEVICE)
            scores = torch.matmul(test_emb, self.cluster_centroid.transpose(-2, -1)) / scale
            attention_weights = torch.softmax(scores, dim=-1)
            return attention_weights
        else:
            raise RuntimeError("No such metric")


class Initialize(nn.Module):
    def __init__(self, num_classes):
        super(Initialize, self).__init__()
        self.num_classes = num_classes

    def get_cluster_centroid(self, test_emb):
        kmeans = KMeans(n_clusters=self.num_classes, n_init=20)
        kmeans.fit_predict(test_emb.cpu().detach().numpy())#聚类预测
        cluster_centers = torch.autograd.Variable(
            #把cluster centers转成pytorch张量，然后转成能够计算梯度的变量
            (torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)).cpu(),
            requires_grad=True)
        return cluster_centers

    def forward(self, test_emb):
        cluster_centroid = self.get_cluster_centroid(test_emb)
        return cluster_centroid
    