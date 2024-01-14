  # -*- coding: utf-8 -*-
'''
@Time : 2023/7/19 10:58
@Author : KNLiu, MQHuang
@FileName : utils.py
@Software : Pycharm
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import anndata as ad
import random
import config
from scipy.stats import entropy
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support, pairwise
from tqdm import tqdm
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def save_checkpoints(model: nn.Module, optimizer: nn.Module, pth: str):
    print("==> Saving Checkpoints")
    checkpoints = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoints, pth)


def load_checkpoints(pth: str):
    print("==> Loading Checkpoints")
    checkpoints = torch.load(pth)
    return checkpoints


def seed_everything(random_state: int):
    '''
    Make the results be reproducible
    :param random_state:用作种子
    :return: None
    '''
    np.random.seed(random_state) #numpy
    random.seed(random_state) #Python内置
    torch.manual_seed(random_state) #pytorch
    torch.cuda.manual_seed_all(random_state) #PyTorch中所有可用GPU


def create_typedict(adata: ad.AnnData):
    # return the indices corresponding to the cell type
    cell_type_list = [types for types in adata.obs["cell.type"].cat.categories]#添加该列中的唯一值
    indices_list = []
    for cell_type in cell_type_list:
        _ = adata.obs["cell.type"] == cell_type #_是一个布尔数组的临时变量
        indices = np.where(_)[0] #找到true的索引。[0]用于提取索引数组的第一个维度，因为np.where返回一个元组，其中第一个维度是满足条件的索引数组。（这里其实不加[0]也可以，不会改变结果）
        indices_list.append(indices)

    cell_type_dict = dict(zip(cell_type_list, indices_list))#zip是匹配，dict再转为字典
    return cell_type_dict


def get_mean(adata: ad.AnnData):
    '''
    calculate the mean representative embedding for each cell type
    :param adata:
    :return: mean_emb
    '''
    index = adata.obs.index
    emb_df = pd.DataFrame(adata.obsm["emb"])
    emb_df.set_index(index, inplace=True)
    mean_emb = emb_df.groupby(by=adata.obs["cell.type"]).mean()#celltype行，emb列的平均矩阵
    mean_emb = torch.tensor(np.asarray(mean_emb), requires_grad=True).to(device=config.DEVICE)
    return mean_emb



def resample(adata: ad.AnnData, proportion, mode):
    '''
    resample according to the match_matrix
    '''
    counts = proportion * adata.X.shape[0]
    counts.sort_index(ascending=True, inplace=True)
    adata.obs["cell.type"] = adata.obs["cell.type"].astype("category")
    cell_type = [types for types in adata.obs["cell.type"].cat.categories]

    resample_cell_indices = []
    for index, value in enumerate(cell_type):
        cell_num = int(counts[index])

        cell_mask = adata.obs['cell.type'] == value
        cell_indices = np.where(cell_mask)[0]

        if len(cell_indices) >= cell_num:
            new_cell_indices = np.random.choice(cell_indices, cell_num,replace=False)
        else:
            new_cell_indices = np.concatenate((cell_indices,
                                              np.random.choice(cell_indices,
                                                               cell_num - len(cell_indices), replace=True)))

        resample_cell_indices.append(new_cell_indices.tolist())

    resample_cell_indices = [element for sublist in resample_cell_indices for element in sublist]
      #将多层嵌套的列表展开为一个单层的列表【              1                  】【              2         】
    adata_resample = adata[resample_cell_indices, :]
    adata_resample.var_names_make_unique()
    adata_resample.obs_names_make_unique()
    print("==> Write the resample annotated data to cache")
    adata_resample.write_h5ad("/volume1/home/mhuang/cellTypeAbundance/data/cache/resample_adata.h5ad")
    #adata_resample.write_h5ad("/Users/mhuang/code/python/abundance/data/cache/resample_adata.h5ad")

    return adata_resample


def attention_score(query, key, emb_size=config.EMB_SIZE):
    '''
    calculate the attention score (Scaled Dot-Product Attention)
    '''
    scale = torch.sqrt(torch.FloatTensor([emb_size])).to(device=config.DEVICE)#缓解梯度消失
    query = query.to(device=config.DEVICE)
    key = key.to(device=config.DEVICE)
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale 
                                 #交换最后两个维度的位置
    attention_weights = torch.softmax(scores, dim=-1) #在最后一个维度进行softmax归一化
    return attention_weights


def get_softproportion(weight, index):
    '''
    平方是为了扩大差异
    '''
    weight = torch.softmax(weight, dim=-1) #dim=-1表示对最后一个维度进行操作; weight.size=(cell,celltype)）
    square_weight = (weight ** 2) / (weight ** 2).sum(dim=1, keepdim=True) #dim=1是横着求和，让她们平方完总和还能等于1；square_weight.size=(cell,celltype)
    # square_weight = F.softmax(weight ** 2)
    ts_proportion = torch.sum(square_weight, 0, keepdim=False) / square_weight.shape[0] #计算按列求和的比例
                    #此时的size变成（1，celltype），区别在于求和时对keepdim的设置
    ts_proportion = ts_proportion.cpu().detach()
    proportion = pd.Series(ts_proportion).sort_index(ascending=True) #这里的ascending好像没有作用
    proportion.index = index #替换指定索引
    return proportion

def get_hardproportion(weight, index):
    weight = torch.softmax(weight, dim=-1)
    pred_indices = weight.argmax(dim=1)
    indices_df = pd.DataFrame(pred_indices.cpu().detach())
    pred_proportion = indices_df.value_counts() / len(indices_df)
                        #计算每个唯一值的出现次数（唯一值变成索引，次数变成值）
    pred_proportion = pred_proportion.sort_index(ascending=True)
    pred_proportion.index = index
    return pred_proportion

# def adjust_proportion(velocity, pred_proportion, ref_proportion, moving_speed: int = 1, beta: float = 0.8):
#     '''
#     adjust proportion according to the difference between pred and ref
#     只在rebalanciMLP的时候会用到（具体是在train.py里的train_MLP）
#     '''
#     # reset index
#     # pred_proportion = pred_proportion.sort_index(ascending=True)
#     # pred_proportion.index = list(range(config.NUM_CLASSES))
#     # ref_proportion = ref_proportion.sort_index(ascending=True)
#     # ref_proportion.index = list(range(config.NUM_CLASSES))
#     gap_proportion = (pred_proportion - ref_proportion).apply(lambda x: 0 if x < 0.005 else x)
#     velocity = beta * gap_proportion + (1 - beta) * velocity 

#     adjust_proportion = pred_proportion + moving_speed * velocity 
#     adjust_proportion = adjust_proportion / adjust_proportion.sum(axis=0)
#     return adjust_proportion, velocity


def adjust(pred_proportion):
    '''
    adjust proportion to prevent too small value
    '''
    adjust_proportion = pred_proportion.apply(lambda x: 0.001 if x < 0.001 else x)#要作用在每个series上
    adjust_proportion = adjust_proportion / adjust_proportion.sum(axis=0)
    return adjust_proportion


def select_cells(scores: torch.Tensor, adata: ad.AnnData, threshold: 0.2, test: bool = True):
    print("==>Select low entropy cells")
    index = adata.obs.index
    #train和test对于select_adata的处理不一样，是因为test数据里是没有cell type的，所以得这样加一个伪标签
    if test:
        p_matrix = F.softmax(scores, dim=1).cpu().detach().numpy()
        celltype_indices = p_matrix.argmax(axis=1)
        celltype_list = [types for types in adata.obs["cell.type"].cat.categories]
        replacement_dict = dict(zip(list(range(config.NUM_CLASSES)), celltype_list))#创建一个字典，将索引值映射为对应的细胞类型
        celltype_series = pd.DataFrame(celltype_indices).replace(replacement_dict) #替换索引
        celltype_series.set_index(index, inplace=True)
        H = entropy(p_matrix, axis=1)
        indices = np.where(H < threshold)[0]
        select_adata = adata[indices]
        select_adata.obs["cell.type"] = celltype_series
        select_adata.obs["cell.type"] = select_adata.obs["cell.type"].astype("category") #转为分类数据类型
    else:
        p_matrix = F.softmax(scores, dim=1).cpu().detach().numpy()
        H = entropy(p_matrix, axis=1)
        indices = np.where(H < threshold)[0]
        select_adata = adata[indices]

    select_adata.var_names_make_unique()
    select_adata.obs_names_make_unique()
    return select_adata


def classify(model: nn.Module, data_loader: DataLoader, num_classes):
    """
    :param model:
    :param train_loader:
    :return: score matrix/distribution matrix
    """
    model.eval() #测试或者验证阶段
    model = model.to(device=config.DEVICE)
    scores = torch.empty((0, num_classes)).to(device=config.DEVICE)#空张量 scores(0, num_classes)
    labels = torch.empty((0, num_classes)).to(device=config.DEVICE)#空张量 scores(0, num_classes)
    
    with torch.no_grad():
        for x, label in data_loader:
            x = x.to(device=config.DEVICE)
            label = label.to(device=config.DEVICE)
            x = x.reshape(x.shape[0], -1)  #展平为一维向量(batch_size, -1)

            score, _ = model(x)
            scores = torch.concat((scores, score), dim=0)
            labels = torch.concat((labels, label), dim=0)
            
    model.train() #训练模式应该是因为后面还要继续训练
    return scores, labels
    

def distance_function(center_matrix):
    total_distance = 0
    num_centers, embedding_size = center_matrix.shape

    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            distance = torch.norm(center_matrix[i] - center_matrix[j])
            total_distance += distance
    average_distance = total_distance / center_matrix.shape[0]

    return average_distance


def loss_func(feat, alpha, cluster_centers):
    '''calculate the kl divergence using test embeddings and cluster centers'''
    q = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha) 
                        #这个函数是用来增加大小为1的维度，用于配合cluster_centers的维度的（否则无法作差）
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()#用t分布来算q，alpha默认是1

    weight = q ** 2 / torch.sum(q, dim=0) 
    p = (weight.t() / torch.sum(weight, dim=1)).t() #归一化来算p

    log_q = torch.log(q) #注意这里额外对q进行了对数化
    loss = F.kl_div(log_q, p, reduction="batchmean")#指对每个批次的kl divergence求平均
    return loss, p


def get_highconfidence(scores: torch.Tensor, adata: ad.AnnData):
    cell_indices = scores.argmax(dim=0)
    indices = [element.item() for element in cell_indices]
    return adata[indices]


class KM_Algorithm:
    """
    1.最大权重匹配
    2.输入的二分图应该是一个经过soft assignment的scores或prob矩阵
    3.二分图是以left：cell，rights：celltype来写的
    """
    def __init__(self, Bipartite_Graph):

        self.Bipartite_Graph = Bipartite_Graph

        # 左右结点数量记录
        self.left = self.Bipartite_Graph.shape[0]  # 以左边（细胞）为主
        self.right_true = self.Bipartite_Graph.shape[1] 
        self.right = self.Bipartite_Graph.shape[1] + self.left
        self.reshape_graph() 

        # step1:最高标准初始化（顶标）
        self.label_left = np.max(self.Bipartite_Graph, axis=1)  # 设置左边顶标为权重最大值（每行的最大值）
        self.label_right = np.zeros(self.right)  # 右边集合的顶标设置为0

        # 初始化辅助变量——是否已匹配
        self.visit_left = np.zeros(self.left, dtype=bool) #全是false
        self.visit_right = np.zeros(self.right, dtype=bool)

        # 初始化右边的匹配结果.如果已匹配就会对应匹配结果
        self.match_right = np.empty(self.right) * np.nan #全是nan

        # 用inc记录需要减去的权值d，不断取最小值故初始化为较大值。权值都为负数，应该不用很大也行
        self.inc = 1000*1000*1000
        self.fail_cell = list()  # 每次匹配重新创建一个二分图匹配对象，所以这个也不用手动重置了

    def reshape_graph(self):
        new = np.ones((self.left, self.left)) * 0 #全0方阵
        self.Bipartite_Graph = np.column_stack((self.Bipartite_Graph, new)) #在右边拼上矩阵
        #new = torch.zeros((self.left, self.left)).to(device='cuda')
        #self.Bipartite_Graph = torch.cat((self.Bipartite_Graph, new),dim=1)
    def match(self, cell):
        cell = int(cell) 
        self.visit_left[cell] = True # 记录下这个cell已经被寻找
        
        for celltype in range(self.right):
            if not self.visit_right[celltype] and self.Bipartite_Graph[cell][celltype] >= 0:    
                  # 如果这个celltype还没访问过       # celltype仍未匹配并且它们之间存在匹配的可能性(不可匹配的点设置为负数，取反后变正数,故正数不可取)
                gap = self.label_left[cell] + self.label_right[celltype] - self.Bipartite_Graph[cell][celltype]  # gap也不会取到不能匹配的那条边
                if gap == 0:   # 差值为0，是可行的替换。所以可以直接尝试替换。后面不行再去将这个一起减去gap。这个列表是记录希望匹配的
                    self.visit_right[celltype] = True
                    # celltype未被匹配，或虽然已被匹配，但是已匹配对象(cell)有其他可选备胎。这里那些是否已访问的列表不需要重置，因为不改变前面的尝试匹配
                    if np.isnan(self.match_right[celltype]) or self.match(self.match_right[celltype]):
                        self.match_right[celltype] = cell# 递归匹配，匹配成功
                        return 1
                # 找到权值最小的差距
                elif self.inc > gap:
                    self.inc = gap  # 等于0的gap不会存在这，所以只要存在可能匹配的情况，gap就不会等于原来
        return 0

    def Kuh_Munkras(self):
        self.match_right = np.empty(self.right) * np.nan
        
        for cell in range(self.left):
            while True:
                self.inc = 1000*1000  # the minimum gap
                self.reset()  # 每次寻找过的路径，所有要重置一下
                # 可找到可行匹配
                if self.match(cell):
                    break #如果返回1的话就会立即break这个while循环
                # 不能找到可行匹配
                # (1)将所有在增广路中的cell方点的label全部减去最小常数
                # (2)将所有在增广路中的celltype方点的label全部加上最小常数
                for k in range(self.left):
                    if self.visit_left[k]:
                        self.label_left[k] -= self.inc
                for n in range(self.right):
                    if self.visit_right[n]:
                        self.label_right[n] += self.inc
        return self.fail_cell

    def calculateSum(self):
        sum = 0
        cells_celltypes = []
        self.fail_cell = [i for i in range(self.left)]
        for i in range(self.right_true):
            if not np.isnan(self.match_right[i]):
                sum += self.Bipartite_Graph[int(self.match_right[i])][i]
                cell_celltype = (int(self.match_right[i]), i)
                cells_celltypes.append(cell_celltype)
                self.fail_cell.remove(int(self.match_right[i]))
         #得到的sum是最短路径
        return cells_celltypes, self.fail_cell
            #匹配成功           #匹配失败
        
    def getResult(self):
        return self.match_right

    def reset(self):
        self.visit_left = np.zeros(self.left, dtype=bool)
        self.visit_right = np.zeros(self.right, dtype=bool)
        
        
def get_KMproportion(weight, index):
    '''如果模型认为细胞是某个细胞类型的概率大于0.5，就直接用argmax转成hard assignment
    反之，则继续用km算法来进行细胞类型的指派'''
    weight = torch.softmax(weight, dim=-1) 
    square_weight = (weight ** 2) / (weight ** 2).sum(dim=1, keepdim=True) #得到初始二分图，权重为扩大差异后的概率 
    
    ## part 1: argmax ##
    index_argmax = []
    index_km = []
    
    for i in range(len(square_weight)):
        if (square_weight[i] > 0.5).any():
            index_argmax.append(i)
        else:
            index_km.append(i)
            
    pred_indices = square_weight[index_argmax].argmax(dim=1)
    argmax_df = pd.DataFrame(pred_indices, index=index_argmax)

    ## part 2: km algorithm ##
    loops = 0
    total_loops = len(index_km)//8 +1
    pred_indices = []
    pred_celltype=[] 
    
    for loops in tqdm(range(total_loops)): 
        km = KM_Algorithm(square_weight[index_km])  #输入二分图
        km.Kuh_Munkras()  # 匹配
        cells_celltypes, _ = km.calculateSum() 

        cells = [index_km[cells[0]] for cells in cells_celltypes] #对应到原始索引的细胞index
        celltypes = [cells[1] for cells in cells_celltypes] #细胞名不需要对应回去
        pred_indices += cells #因为是列表
        pred_celltype += celltypes
        index_km = [x for x in index_km if x not in cells]

    print("==> KM algortithm over!")
    km_df = pd.DataFrame(pred_celltype, index=pred_indices)

    ## part 3: prediction results ##
    pred_df = pd.concat([argmax_df,km_df],axis=0)
    pred_df = pred_df.sort_index(ascending=True) #细胞数量行1列的矩阵，行索引是细胞index，值是预测的label
    proportion = pd.DataFrame(pred_df[0].value_counts()/len(pred_df))
    proportion = proportion.sort_index(ascending=True) #细胞类型数量行1列的矩阵
    proportion.index = index
    proportion = pd.Series(proportion.iloc[:,0])#不然后面有些函数还得分类讨论
    return proportion, pred_df


def metrics(scores, labels, mode, pred_labels):
    '''
    evaluate the prediction results
    soft assignment对于单个细胞预测的准确率还是得利用hard assignment，因为它跳过了对单个细胞的预测、直接预测比例
    '''
    dict_correct = {i: 0 for i in range(config.NUM_WORKERS)}
    dict_samples = {i: 0 for i in range(config.NUM_WORKERS)}
    y_true, y_pred=[], []
    cell_type = [i for i in range(config.NUM_WORKERS)]

    if mode == 'soft' or 'hard':
        _, prediction = scores.max(dim=1) #predictions是(cell_num,1)的矩阵（张量的写法是(cell_num,)）
        _, indices = labels.max(dim=1)#_是一个占位符，第一个是样本的最大值，第二个是其索引
    elif mode == 'km':
        if pred_labels is None:
            raise ValueError("The pred_labels is required!") 
        else:
            prediction = pred_labels
            _, indices = labels.max(dim=1)
    else:
        raise ValueError("Invalid mode!")
        
    for i in range(prediction.size(0)):
        m = indices[i].item() 
        n = prediction[i].item()
        y_true.append(m)
        y_pred.append(n)
        if m in dict_correct:
            dict_samples[m] += 1
            if m == n:
                dict_correct[m] += 1
        else:
            print("The number of cell type is not suitable!")
                
    #（1）每个cell type的指标
    p,r,f,num_true=precision_recall_fscore_support(y_true=y_true,y_pred=y_pred,labels=cell_type,average=None)
    precision = pd.DataFrame(p)  
    recall = pd.DataFrame(r)
    
    #（2）整体的指标
    f1_score_micro = f1_score(y_true, y_pred,labels=cell_type,average='micro')
    f1_score_macro = f1_score(y_true, y_pred,labels=cell_type,average='macro')
    f1score = pd.DataFrame([f1_score_micro,f1_score_macro], index=['f1score_micro','f1score_macro'])   
    
    num_correct = sum(dict_correct.values())
    num_samples = sum(dict_samples.values())
    accuracy = float(num_correct) / float(num_samples)
    print(f"Got {num_correct} / {num_samples} with accuracy' {accuracy * 100:.2f}%")
        
    return accuracy, f1score, precision, recall


class KMM:
    '''Kernel mean matching'''
    def __init__(self, gamma=1.0, B=1.0, eps=None):
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = pairwise.rbf_kernel(np.asarray(Xs), None, self.gamma) #源域数据的核矩阵（单位阵？）
        kappa = np.sum(pairwise.rbf_kernel(Xs, Xt, self.gamma)* float(ns) / float(nt), axis=1) #源域和目标域数据之间的交叉核矩阵（全0矩阵？）

        K = matrix(K.astype('float64')) #原来是float32
        kappa = matrix(kappa.astype('float64'))
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)]) #约束条件
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h, options={'show_progress': False})
        beta = np.array(sol['x']) #是一个向量
        return beta

    def reweight(self, adata_reweight, adata_test):
        cell_type = [i for i in adata_test.obs['cell.type'].cat.categories]
        reweight_X = []
        for cell in cell_type:
            train_cache = adata_reweight[adata_reweight.obs['cell.type']==cell]
            test_cache = adata_test[adata_test.obs['cell.type']==cell]
            Xs = train_cache.X
            Xt = test_cache.X
            beta = self.fit(Xs, Xt)
            Xs_adjust = Xs * beta
            reweight_X.append(Xs_adjust)
        reweight_X = np.concatenate(reweight_X, axis=0)
        adata_reweight.X = reweight_X

        print("==> Write the reweighted data to cache")
        adata_reweight.write_h5ad("/volume1/home/mhuang/cellTypeAbundance/data/cache/resample_adata.h5ad")
        #adata_reweight.write_h5ad("/Users/mhuang/code/python/abundance/data/cache/resample_adata.h5ad")

        return adata_reweight
    

def plot_loss(loss_s,loss_t, min_dataloader):
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    
    loss_s.reset_index(drop=True, inplace=True)
    loss_t.reset_index(drop=True, inplace=True)
    loss_avg = loss_s.groupby(loss_s.index // min_dataloader).mean()
    loss_avg = pd.concat([loss_avg,loss_t],axis=1)
    loss_avg['epoch'] = range(1, len(loss_avg)+1)
    
    num = 0
    for column in loss_avg.drop('epoch', axis=1):
        num += 1
        plt.plot(loss_avg['epoch'], loss_avg[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
    for i in range(config.EPOCHS, len(loss_avg['epoch']) + 1, config.EPOCHS):
        plt.axvline(x=i, color='black', linestyle='--', alpha=0.5)

    plt.legend(loc=2, ncol=2)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(range(min(loss_avg['epoch']), max(loss_avg['epoch']) + 1, 1))
    plt.show()
    
    
def KS_test():
    '''不对，应该放到reweight里，如果分布一致就不用再reweight了；如果放到early stop里的话就，它的adata.X是不会变的；或者说调整early stop的方式，一直运行到分布一致为止'''
    _, pvalues= ks_2samp(data0, data1)
    if pvalues>0.05: #拒绝H0,分布一致
        return