import torch
import torch.nn as nn
import torch.optim as optim
import config
import pandas as pd
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
from model import MLP, Initialize
from tqdm import tqdm
from train import train_MLP
from utils import classify, resample, get_mean, plot_loss
from utils import load_checkpoints, seed_everything, save_checkpoints, attention_score
from utils import get_hardproportion, adjust, get_softproportion, loss_func
from dataset import scDataset
from utils import get_KMproportion, metrics, KMM

class propDEC(nn.Module):
    def __init__(self, input_size, num_classes, metric, mode):
        super(propDEC, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.metric = metric
        self.mode = mode
        self.mlp = MLP(input_size=self.input_size, num_classes=self.num_classes)

    def create_dataset(self, dataset_name, file_name, mode) -> Dataset:
        self.file_name = file_name
        self.dataset_name = dataset_name
        assert mode == "train" or mode == "test"
        if mode == "train":
            dataset = scDataset(dataset_name, file_name, train=True)
            self.ref_dataset = dataset
            self.ref_adata = dataset.train_adata
            self.cell_type = [types for types in self.ref_adata.obs["cell.type"].cat.categories]
            ref_proportion = self.ref_adata.obs["cell.type"].value_counts() / len(self.ref_adata)
            ref_proportion = ref_proportion.sort_index(ascending=True)
            ref_proportion.index = self.cell_type
            self.ref_proportion = ref_proportion.rename("ref")
            self.train_dataloader = DataLoader(self.ref_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)
        elif mode == "test":
            dataset = scDataset(dataset_name, file_name, train=False)
            self.test_dataset = dataset
            self.test_adata = dataset.test_adata
            self.cell_type = [types for types in self.test_adata.obs["cell.type"].cat.categories]
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=config.BATCH_SIZE,
                                              num_workers=config.NUM_WORKERS, shuffle=True)
        else:
            raise RuntimeError("No such mode")

        return dataset

    def forward(self, epochs, load=True, alpha=1.0):
        self.proportion = pd.DataFrame(torch.randn(size=(self.num_classes, 0)),
                                       index=self.cell_type)
        self.proportion = pd.concat((self.proportion, self.ref_proportion), axis=1)
        self.accuracy = pd.DataFrame(torch.randn(size=(self.num_classes, 0)),index=self.cell_type)
        self.f1scores = pd.DataFrame(torch.randn(size=(2, 0)), index=['f1score_micro','f1score_macro'])
        self.precision = pd.DataFrame(torch.randn(size=(self.num_classes, 0)),index=self.cell_type)
        self.recall = pd.DataFrame(torch.randn(size=(self.num_classes, 0)),index=self.cell_type)
        self.loss_s = pd.DataFrame(columns=['source'])
        self.loss_t = pd.DataFrame(columns=['target'])
        
        sc.settings.verbosity = 0
        
        # initialize model and optimizer
        model = self.mlp.to(device=config.DEVICE)

        # pretrain mlp model with true labels
        model = train_MLP(model, self.train_dataloader, load=False) 
   
        init = Initialize(num_classes=self.num_classes)
        test_emb = self.mlp.extract_emb(self.test_dataloader.dataset.test_adata)
        cluster_centroid = init(test_emb)
        
        # initialize model and optimizer
        optimizer = optim.Adam(list(model.parameters()) + [cluster_centroid],
                               lr=config.CLASSIFY_LEARNING_RATE,
                               betas=(0.1, 0.999))
        # loss
        classify_criterion = nn.CrossEntropyLoss()
        l2_criterion = nn.MSELoss()

        # seed everything
        seed_everything(config.SEED)

        # weight initialization
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0) #把bias初始化为0更有助于学到合适的bias

        for epoch in range(epochs):
            if epoch == 0:
                train_dataloader = self.train_dataloader
            else:
                resample_dataset = scDataset("cache_resample", "resample_adata.h5ad", train=True)
                train_dataloader = DataLoader(resample_dataset, batch_size=config.BATCH_SIZE,
                                              num_workers=config.NUM_WORKERS, shuffle=True)
            # seed everything
            seed_everything(42)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                      max_lr=0.001,
                                                      steps_per_epoch=len(train_dataloader),
                                                      epochs=config.EPOCHS)

            identity_matrix = torch.eye(self.num_classes).to(device=config.DEVICE)

            # start training process
            if load and epoch != 0:
                checkpoints = load_checkpoints(config.CLASSIFIER_CHECKPOINT)
                model.load_state_dict(checkpoints["model"])
                optimizer.load_state_dict(checkpoints["optimizer"])

            for train_epoch in range(config.EPOCHS):
                loop = tqdm(enumerate(train_dataloader), leave=False, total=len(train_dataloader))
                # velocity = 0
                for idx, (x, label) in loop:
                    # data
                    x = x.to(device=config.DEVICE)  # Get the data to cuda is possible
                    label = label.to(device=config.DEVICE).squeeze()#确保label的维度符合预期
                    # reshape
                    x = x.reshape(x.shape[0], -1)
                    label = label.reshape(label.shape[0], -1)

                    # mlp forward
                    x_hat, x_emb = model(x)
                    loss_mlp = classify_criterion(x_hat, label)

                    # dec forward
                    loss_dec, _ = loss_func(x_emb, alpha, cluster_centroid)
                    attention_matrix = attention_score(cluster_centroid, cluster_centroid)#距离方阵
                    attention_matrix.cpu().detach() 
                    loss_dec += 10 * l2_criterion(attention_matrix, identity_matrix)

                    loss = loss_mlp + loss_dec
                    loss_s_df = pd.DataFrame({'source': [loss_mlp.detach().numpy()]})#单个batch
                    self.loss_s = pd.concat([self.loss_s, loss_s_df])
                    
                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loop.set_description(
                        f"[Epoch {train_epoch + 1}/{config.EPOCHS}]")
                    loop.set_postfix(loss=loss.item())

                    scheduler.step()

                if config.SAVE_MODEL and (train_epoch + 1) % 5 == 0:
                    save_checkpoints(model, optimizer, pth=config.CLASSIFIER_CHECKPOINT)
                    
                # scores_test, labels_test = classify(model, self.test_dataloader, self.num_classes)
                # loss_t = classify_criterion(scores_test, labels_test)
                # loss_t_df = pd.DataFrame({'target':[loss_t.detach().numpy()]}) #所有batch
                # self.loss_t = pd.concat([self.loss_t,loss_t_df])

            scores_test, labels_test = classify(self.mlp, self.test_dataloader, self.num_classes)

            # proportion
            pred_labels = None
            if self.mode == "soft":
                pred_proportion = get_softproportion(scores_test, self.cell_type)
            elif self.mode == "km":
                pred_proportion, pred_labels = get_KMproportion(scores_test, self.cell_type)
            else:
                pred_proportion = get_hardproportion(scores_test, self.cell_type)

            pred_proportion = adjust(pred_proportion)
            pred_proportion = pred_proportion.rename(f"predict_{epoch + 1}")
            
            # evaluation metrics
            _, f1scores, precision, recall = metrics(scores_test,labels_test, self.mode, pred_labels)
            precision.index = self.cell_type
            recall.index = self.cell_type
            
            self.proportion = pd.concat((self.proportion, pred_proportion.round(decimals=4)), axis=1)
            self.f1scores = pd.concat((self.f1scores, f1scores), axis=1)
            self.precision = pd.concat((self.precision, precision), axis=1)
            self.recall = pd.concat((self.recall, recall), axis=1)
            
            if epoch != 0 and (self.proportion.iloc[:, -1] - self.proportion.iloc[:, -2]).abs().sum() < 0.01:
                print("Early Stopping")
                break
            print("==> resample")
            adata_resample = resample(self.ref_adata, self.proportion.iloc[:, -1], mode=self.mode)
            k = KMM()
            k.reweight(adata_resample, self.test_adata)
        print("==> Finish!")
        
        #plot_loss(self.loss_s, self.loss_t,len(train_dataloader))

        #规范一下最后的输出结果
        ground_truth = self.test_adata.obs.groupby('cell.type').count()/len(self.test_adata)
        ground_truth = ground_truth.iloc[:,0]
        ground_truth.name='Ground_truth'
        self.proportion = pd.concat([self.proportion,ground_truth],axis=1)

        columns = [f"predict_{epoch + 1}" for epoch in range(self.f1scores.shape[1])]
        self.f1scores.columns = columns
        self.precision.columns = columns
        self.recall.columns = columns
        
        return scores_test, self.proportion, self.f1scores, self.precision, self.recall

    def extract_emb(self, data_loader):
        '''
        The embedding is stored in adata.obsm["emb"]
        在model里的mlp模型也设计了一个extract_emb函数，但是它输入的是adata，就不需要像这样一样，一直迭代
        （因为数据太大的时候，输入adata会让cuda超出内存）
        '''
        new_dataloader = DataLoader(data_loader.dataset,
                                    batch_size=config.BATCH_SIZE,
                                    num_workers=config.NUM_WORKERS,
                                    shuffle=False)
        embs = torch.empty(size=(0, config.EMB_SIZE)).to(device=config.DEVICE)
        model = self.mlp.to(device=config.DEVICE)

        with torch.no_grad():
            for x, _ in new_dataloader:
                x = x.to(device=config.DEVICE)
                x = x.reshape(x.shape[0], -1)

                _, emb = model(x)
                embs = torch.concat((embs, emb), dim=0)
        if data_loader.dataset.train:
            data_loader.dataset.train_adata.obsm["emb"] = embs.cpu().detach().numpy()
        else:
            data_loader.dataset.test_adata.obsm["emb"] = embs.cpu().detach().numpy()

        return embs
