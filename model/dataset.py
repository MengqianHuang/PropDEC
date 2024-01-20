# -*- coding: utf-8 -*-
'''
@Time : 2023/7/19 10:58
@Author : KNLiu, MQHuang
@FileName : dataset.py
@Software : Pycharm
'''
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import os
import torch
import config
from torch.utils.data import Dataset
from typing import Any, Tuple


# ---- some functions for processing data
def process_adata(adata, prefix, result_dir, min_genes=10, min_cells=10, celltype_label="cell.type"):
    '''Procedures for filtering single-cell data
       1. Filter low-quality cells and genes;
       2. Filter nonsense genes;
       3. Normalize and log-transform the data;
       4. Change all gene names into UPPER;
       5. Remove cells with no labels;
    '''
    adata.var_names = [i.upper() for i in list(adata.var_names)]  # avoid some genes having lower letter

    # make names unique
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # 3 prefilter_specialgene: MT and ERCC  -> from ItClust package
    gene1pattern = "ERCC"
    gene2pattern = "MT-"
    id_tmp1 = np.asarray([not str(name).startswith(gene1pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(gene2pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)

    # handel exception when there are not enough cells or genes after filtering
    if adata.shape[0] < 3 or adata.shape[1] < 3:
        return None

    # Get depth
    get_depth(adata, result_dir, prefix)

    # 4 normalization,var.genes,log1p
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=10000)
    # total count equal to the median of the counts_per_cell
    sc.pp.log1p(adata)

    # cells with cell types
    cells = adata.obs.dropna(subset=[celltype_label]).index.tolist()
    adata = adata[cells]
    return adata


def get_depth(adata, result_dir, prefix):
    '''Save distance as txt file
    '''
    depth1 = adata.X.sum(axis=1)

    np.savetxt(result_dir + os.sep + prefix + "_Depth.txt", depth1)


class scDataset(Dataset):
    def __init__(self, dataset_name: str, *args, data_root: str = config.DATA_ROOT,
                 train: bool = True):
        super(scDataset, self).__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.train = train
        
        if dataset_name =='cache_resample':
            data_dir = config.CACHE_PATH
        else:
            data_dir = os.path.join(data_root, dataset_name)
            
        if not os.path.exists(data_dir):
            raise RuntimeError("Warning: No such file or directory")
        else:
            if train:
                train_data_dir = os.path.join(data_dir, args[0])
                if os.path.exists(train_data_dir):
                    self.train_adata = sc.read(train_data_dir)
                    train_label_df = pd.DataFrame(self.train_adata.obs["cell.type"])
                    self.train_label = pd.get_dummies(train_label_df)
                    self.train_label = np.array(self.train_label)
                else:
                    raise RuntimeError("Warning: No such train data file")
            else:
                test_data_dir = os.path.join(data_dir, args[0])
                if os.path.exists(test_data_dir):
                    self.test_adata = sc.read(test_data_dir)
                    test_label_df = pd.DataFrame(self.test_adata.obs["cell.type"])
                    self.test_label = np.array(pd.get_dummies(test_label_df))
                else:
                    raise RuntimeError("Warning: No such test data file")

    def adata(self, train: bool = True):
        if train:
            return self.train_adata
        else:
            return self.test_adata

    def __len__(self) -> int:
        if self.train:
            return self.train_adata.X.shape[0]
        else:
            return self.test_adata.X.shape[0]

    def __getitem__(self, index) -> Tuple[Any, Any]:
        if self.train:
            data = torch.tensor(self.train_adata[index].X)
            label = torch.tensor(self.train_label[index], dtype=torch.float)
        else:
            data = torch.tensor(self.test_adata[index].X)
            label = torch.tensor(self.test_label[index], dtype=torch.float)

        return data, label


class Process:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def preprocess(self, adata: ad.AnnData):
        # preprocess
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=3000, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]

        sc.pp.scale(adata)

        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        return adata

    def split_train_test(self, adata: ad.AnnData, train_proportion: list, test_proportion: list):
        self.train_proportion = train_proportion
        self.test_proportion = test_proportion
        assert int(sum(self.train_proportion)) == 1 and int(sum(self.test_proportion)) == 1

        cell_type = [types for types in adata.obs["cell.type"].cat.categories]
        total_num = len(adata) #/ 2  
        train_total_num = total_num * 0.7
        test_total_num = total_num * 0.3
        train_cell_num = [train_total_num * proportion for proportion in self.train_proportion]  # a list
        test_cell_num = [test_total_num * proportion for proportion in self.test_proportion]
        adata.obs["dataset_type"] = "type"

        for index, value in enumerate(cell_type):
            cell_mask = adata.obs['cell.type'] == value
            cell_indices = np.where(cell_mask)[0]
            train_num = int(train_cell_num[index])
            test_num = int(test_cell_num[index])
            cell_num = train_num + test_num

            if len(cell_indices) >= cell_num:
                train_indices = np.random.choice(cell_indices, train_num, replace=False) #无放回抽样
                remaining_indices = np.setdiff1d(cell_indices, train_indices)
                test_indices = np.random.choice(remaining_indices, test_num, replace=False)

                adata.obs['dataset_type'][train_indices] = 'train'
                adata.obs['dataset_type'][test_indices] = 'test'
            else:
                train_indices = np.random.choice(cell_indices, train_num, replace=True) #有放回抽样
                remaining_indices = np.setdiff1d(cell_indices, train_indices)
                test_indices = np.random.choice(remaining_indices, test_num, replace=True)

                adata.obs['dataset_type'][train_indices] = 'train'
                adata.obs['dataset_type'][test_indices] = 'test'
        
        train_type = adata.obs['dataset_type'] == 'train'
        train_adata = adata[train_type, :]
        test_type = adata.obs['dataset_type'] == 'test'
        test_adata = adata[test_type, :]

        train_adata.write(f'/volume1/home/mhuang/cellTypeAbundance/data/{self.dataset_name}/{self.dataset_name}_train.h5ad')
        test_adata.write(f'/volume1/home/mhuang/cellTypeAbundance/data/{self.dataset_name}/{self.dataset_name}_test.h5ad')

        return train_adata, test_adata

def FindMarkers(adata_test_pre:ad.AnnData, adata_ref_pre:ad.AnnData, top_n_genes=1500)
    adata = adata_test_pre.concatenate(adata_ref_pre, batch_key="dataset_type")
    
    # preprocess
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata,target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    adata_ref = adata[adata.obs['dataset_type']=='1']
    adata_test = adata[adata.obs['dataset_type']=='0']
    
    # select top_n marker genes
    sc.tl.rank_genes_groups(adata_ref,groupby='cell.type',use_raw=False,method="wilcoxon") #method='t-test','wilcoxon'
    sc.tl.filter_rank_genes_groups(adata_ref,groupby="cell.type",min_in_group_fraction=0.8,max_out_group_fraction=0.2) 

    markers=sc.get.rank_genes_groups_df(adata_ref, group=adata_ref.obs['cell.type'].unique())
    markers=markers.loc[~ markers.names.isna()]
    markers['abs_score']=markers.scores.abs()
    markers.sort_values('abs_score',ascending=False,inplace=True)
    markers = markers.groupby('group').head(top_n_genes).sort_values('group')['names'].unique()
    print("There exists unique marker genes: "+len(markers))
    
    adata_ref = adata_ref[:,markers]
    adata_test = adata_test[:,markers]
    adata_ref.obs.rename(columns={'cellType': 'cell.type'}, inplace=True)
    adata_test.obs.rename(columns={'cellType': 'cell.type'}, inplace=True)
    
    return adata_ref, adata_test
