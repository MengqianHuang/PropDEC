import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np

import anndata as ad
def add_celltype(adata:ad.AnnData, group:list, meta:pd.DataFrame):
    '''
    处理Haber等人的数据。
    这样使用：group = ['B1','B2','B3','B4','B9','B10']
            adata_1 = add_celltype(adata, group, meta)

    '''
    cell_indeies_list = []
    celltype_meta =pd.DataFrame()
    num = 0
    
    for i in range(len(group)):
        meta_group = meta[meta['group']==group[i]]
        meta_group_list = meta_group['cell.name'].tolist()
        cell_group = adata[adata.obs['group']==group[i]].obs.index.str.split('-')
        celltype_meta_group = meta['cell.type'][meta['group']==group[i]]
        celltype_meta = pd.concat([celltype_meta, celltype_meta_group],axis=0)

        meta_index_list=[]
        cell_index_list=[]
        for j in range(len(cell_group)):
            if cell_group[j][0] in meta_group_list:
                meta_index = meta_group_list.index(cell_group[j][0])
                meta_index_list.append(meta_index)
                cell_index_list.append(j)
        
        #检查这种偷懒做法有没问题(前面的计算假定所有细胞都是按首字母排序的)
        meta_index_list2 = sorted(meta_index_list)
        cell_index_list2 = sorted(cell_index_list)
        if (meta_index_list != meta_index_list2)or(cell_index_list != cell_index_list2):
            print("==> 第 "+group[i]+" 组得重新写的复杂点.")
        else:
            if i==0:
                lst = [m for m in cell_index_list]
            else:
                lst = [m+num for m in cell_index_list]
            cell_indeies_list += lst

        num += len(cell_group)

    #把结果拼一起
    adata = adata[cell_indeies_list,:]
    celltype_meta.index = adata.obs.index
    adata.obs['cell.type']=celltype_meta
    
    return adata
