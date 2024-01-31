library(dplyr)
library(DCATS)
library(Seurat)

# 1. data preprocess 
ppms_all <- Read10X(data.dir = "/Users/mhuang/code/python/abundance/data/GSE144744/RNA_counts")
meta <- read.csv("/Users/mhuang/code/python/abundance/data/GSE144744/meta.csv",header = TRUE)
ppms_all <- CreateSeuratObject(counts = ppms_all, project = "ppms", min.cells = 3, 
                                min.features = 200, meta.data = meta)
ppms_all <- readRDS("/Users/mhuang/code/python/abundance/data/GSE144744/seurat_all.rds")

selected_cells_logical <- ppms_all@meta.data$cluster_names %in% c('T06', 'T07', 'T09', 'T10')
ppms <- ppms_all[, selected_cells_logical]
selected_cells_logical <- ppms$group %in% c('PPMS')
ppms <- ppms[, selected_cells_logical]
rm(ppms_all,selected_cells_logical)

# 2.standard process 
ppms <- NormalizeData(ppms)
ppms <- ppms %>%
  ScaleData() %>%
  RunPCA() %>%
  RunUMAP(dims = 1:30) %>%
  FindNeighbors()
  
UMAPPlot(ppms, group.by = "cluster_names")

# 3.composition prediction 
## 1) get 4 clusters
ppms.new<-FindClusters(ppms,resolution = 0.13)
UMAPPlot(ppms.new, group.by = "seurat_clusters")

## 2)correct the prediction
similarity_mat = knn_simMat(ppms.new@graphs$RNA_snn, ppms.new$seurat_clusters)
count_mat = table(ppms.new$sample, ppms.new$seurat_clusters)
count_mat_new <- count_mat
for (i in seq_len(nrow(count_mat))) {
  count_mat_new[i, ] <- sum(count_mat[i, ]) * multinom_EM(count_mat[i, 
  ], similarity_mat)$mu
}
count_mat_new

## 3) evaluation
prop_pred<-colSums(count_mat_new)/24153
real_values <- c(0.514594, 0.250238, 0.215667, 0.019501)
sqrt(mean((prop_pred - real_values)^2)) #等于0.1115789
