## install dependencies
install.packages(c("MCMCpack", "matrixStats", "robustbase", "aod", "e1071"))
## dependencies for vignette
install.packages(c("SeuratObject", "Seurat", "robustbase", "aod", "e1071"))
devtools::install_github('satijalab/seurat-data')
# install.packages("devtools")
devtools::install_github("holab-hku/DCATS")

library(DCATS)

### 1.github quick start
data("simulation")
#data("Haber2017")
knn_mat = knn_simMat(simulation$knnGraphs, simulation$labels)
sim_count = rbind(simulation$numb_cond1, simulation$numb_cond2)
sim_design = data.frame(condition = c("c1", "c1", "c2"))
knn_mat[colnames(sim_count),]
res = dcats_GLM(as.matrix(sim_count), sim_design, similarity_mat = knn_mat)
print(res$LRT_pvals)

### 2. github tutorial
### https://htmlpreview.github.io/?https://github.com/linxy29/DCATS_anlysis/blob/master/vignette/Integrate_with_seurat.html
library(Seurat)
library(SeuratData)
library(patchwork)
#BiocManager::install("SeuratData",force = TRUE)
library(tidyverse)
library(stringr)

#SeuratData::InstallData("ifnb")
ifnb <- LoadData("ifnb")

# split the dataset into a list of two seurat objects (stim and CTRL)
ifnb.list <- SplitObject(ifnb, split.by = "stim")

# normalize and identify variable features for each dataset independently
ifnb.list <- lapply(X = ifnb.list, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})

# select features that are repeatedly variable across datasets for integration
features <- SelectIntegrationFeatures(object.list = ifnb.list)

# perform integration（用cca把来自不同条件的矩阵整合到一起，注意features也降了）
immune.anchors <- FindIntegrationAnchors(object.list = ifnb.list, anchor.features = features)
immune.combined <- IntegrateData(anchorset = immune.anchors)

# perform an integrated analysis
DefaultAssay(immune.combined) <- "integrated"

# Run the standard workflow for visualization and clustering
immune.combined <- ScaleData(immune.combined, verbose = FALSE)
immune.combined <- RunPCA(immune.combined, npcs = 30, verbose = FALSE)
immune.combined <- RunUMAP(immune.combined, reduction = "pca", dims = 1:30)

immune.combined <- FindNeighbors(immune.combined, reduction = "pca", dims = 1:30)
immune.combined <- FindClusters(immune.combined, resolution = 0.5)

#differential analysis
##该算法必须需要样本名（在实际数据中就是指rep1，rep2这种的）才能运行
sampleID = str_c(immune.combined$stim,sample(c("s1", "s2"), length(Idents(immune.combined)), replace = TRUE))
immune.combined = AddMetaData(immune.combined, sampleID, 'sample')

# Visualization
DimPlot(immune.combined, reduction = "umap", group.by = "sample")

#similarity matrix
knn_mat = knn_simMat(immune.combined@graphs$integrated_snn, immune.combined$seurat_annotations)
#cell type proportion
count_mat = table(immune.combined$sample, immune.combined$seurat_annotations)
#design matrix
design_mat = data.frame(condition = c("CTRL", "CTRL", "STIM", "STIM"))
dcats_GLM(count_mat, design_mat, similarity_mat = knn_mat)
