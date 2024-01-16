options(repos = c(CRAN = "https://mirrors.ustc.edu.cn/CRAN/"))
devtools::install_github('kharchenkolab/cacoa')

BiocManager::install(c("clusterProfiler", "DESeq2", "DOSE", "EnhancedVolcano", "enrichplot", "fabia", "GOfuncR", "Rgraphviz"))
devtools::install_github("kharchenkolab/sccore", ref="dev")
library(cacoa)
install.packages("candisc")
library(rgl)

install.packages("rgl",type="source")
install.packages("heplots",type="binary")


library(candisc)
getPValueDf <- function(cao, cell.group.order) {
  freqs <- cao$test.results$coda$cnts %>% {. / rowSums(.)}
  pval.df <- cao$sample.groups %>% {split(names(.), .)} %>% 
    {matrixTests::col_wilcoxon_twosample(freqs[.[[1]],], freqs[.[[2]],])} %$% 
    setNames(pvalue, rownames(.)) %>% 
    p.adjust("BH") %>% cacoa:::pvalueToCode(ns.symbol="") %>% 
    {tibble(ind=names(.), freq=., coda=cao$test.results$coda$padj[names(.)])} %>% 
    mutate(ind=factor(ind, levels=cell.group.order), coda=cacoa:::pvalueToCode(coda, ns.symbol="")) %>% 
    rename(Freqs=freq, CoDA=coda)
  
  return(pval.df)
}

addPvalueToCoda <- function(gg, cao, x.vals, show.legend=FALSE, size=4, legend.title="Significance") {
  pval.df <- getPValueDf(cao, cell.group.order=levels(gg$data$ind))
  gg <- gg + 
    geom_text(aes(x=x.vals[1], label=CoDA, color="CoDA"), data=pval.df, vjust=0.75, size=size) +
    geom_text(aes(x=x.vals[2], label=Freqs, color="Wilcox"), data=pval.df, vjust=0.75, size=size) +
    scale_color_manual(values=c("black", "darkred"))
  
  if (show.legend) {
    gg <- gg + 
      cacoa:::theme_legend_position(c(1, 0.04)) +
      guides(fill=guide_none(), color=guide_legend(title=legend.title))
  }
  return(gg)
}

## simulate the data
n.cell.types <- 7
cell.types <- paste('type', 1:n.cell.types, sep='')

n.samples <- 20  # number of samples in one group
groups.name <- c('case', 'control')
groups.type <- c(rep(groups.name[1], n.samples), rep(groups.name[2], n.samples))
sample.names <- paste(groups.type, 1:(2*n.samples), sep = '')
groups <- setNames(groups.type %in% groups.name[1], sample.names)

palette <- RColorBrewer::brewer.pal(n.cell.types, "Set1") %>% setNames(cell.types)#随机取色

sample_groups <- c("control", "case")[groups + 1] %>% setNames(names(groups))

sg_pal <- c(case="#BF1363", control="#39A6A3")

cnt.shift <- 100
cnt.shift2 <- -15

set.seed(1124)

cnts <- lapply(1:(2 * n.samples), function(i) round(rnorm(n.cell.types, mean=50, sd=5))) %>% 
  do.call(rbind, .) %>% set_rownames(sample.names) %>% set_colnames(cell.types)#生成正态分布的随机数

cnts[,1] <- cnts[,1] + groups * cnt.shift
cnts[,2] <- cnts[,2] + groups * cnt.shift2
cnts[,3] <- cnts[,3] + groups * cnt.shift2

freqs <- cnts %>% {. / rowSums(.)}

res <- cacoa:::runCoda(cnts, groups, n.seed=239)
dfs <- cacoa:::estimateCdaSpace(cnts, groups)
