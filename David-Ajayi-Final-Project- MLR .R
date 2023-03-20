library(readr)
library(tidyverse)
library(GGally)
library(e1071)
library(MASS)
library(caret)




ds.obj <- read.csv("insurance.csv", sep=",", header=TRUE)

dataset.obj <- as.data.frame(ds.obj)
dataset.obj$smoker <- as.factor(dataset.obj$smoker)

summary(dataset.obj)
str(dataset.obj)

View (dataset.obj)

dataset.obj2 <- dataset.obj[,-2][,-5][,-6]


view (dataset.obj2)

ggpairs(dataset.obj2) ## produce a matrix of all bivariate plots and correlations


cor(dataset.obj2) ## produce a matrix of all correlations



nzv <- nearZeroVar(dataset.obj, saveMetrics = TRUE)
## assert that there are no variables with zero or nearly-zero variation
all(nzv$zeroVar == FALSE) && all(nzv$nzv == FALSE)
nzv

set.seed(2001) ## set seed because we're creating training+testing and hold-out sets via a random-draw.
trainIndices <- createDataPartition(dataset.obj$charges, ## indicate which var. is outcome
                                    p = 0.8, # indicate proportion to use in training-testing
                                    list = FALSE, 
                                    times = 1)

training <- dataset.obj[trainIndices,]
holdout <- dataset.obj[-trainIndices,]

## centering and scaling as part of the pre-processing step
preProcValues <- preProcess(training, method = c("center", "scale"))

## Next, create the scaled+centered of the training+testing subset of the dataset
trainTransformed <- predict(preProcValues, training) 
## apply the same scaling and centering on the holdout set, too
holdoutTransformed <- predict(preProcValues, holdout)



fitControl <- trainControl(
  method = "repeatedcv", ## perform repeated k-fold CV
  number = 5, ## 10-fold CV
  ## repeated ten times
  repeats = 3)


#Fitting the MLR Regression

grid <- expand.grid(lambda = 10 ^ seq(10, -2, length = 100),
                    alpha = 1)

lassofit <- train(charges ~ ., data = trainTransformed, 
                 method = "lm",
                 ## alpha = 0, ## indicating MLR regression
                 trControl = fitControl, 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 )

lassofit

names(lassofit)

plot(lassofit)

predvals <- predict(lassofit, holdoutTransformed)
postResample(pred = predvals, obs = holdoutTransformed$charges)

varImp(lassofit)
