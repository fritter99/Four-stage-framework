library(synthpop)
library(xlsx)
data <- read.xlsx("Experimental dataset.xlsx",sheetName =1)[c(1:91),c(1:11)]
res _0<- syn(data,m=1)
res _1<- syn(data,m=2)
res _2<- syn(data,m=3)
res _3<- syn(data,m=4)

data <- read.xlsx("Synthetic dataset.xlsx",sheetName =1)[c(1:273),c(2:11)]
mycor <- cor(data,method='pearson')
corrplot(mycor,method='ellipse',type='upper',addCoef.col = 'black')