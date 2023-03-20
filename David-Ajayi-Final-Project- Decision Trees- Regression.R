library(readr)
library(tidyverse)
library(GGally)
library(caret)
library(MASS)
library(randomForest)
library(gbm)
library(caret)
library(doParallel)

cl <- makePSOCKcluster(5)
registerDoParallel(cl)


insurance <- read_csv("C:/Users/donja/OneDrive/Desktop/insurance.csv") #load the insurance dataset
data(insurance) ## load the Auto dataset 
attach(insurance) ## attach it to make its variables accesible directly


ds.obj <- as.data.frame(insurance)
summary(ds.obj)

View(insurance) ## View the dataset
summary(insurance)

glimpse(ds.obj)


insurance$smoker <- as.factor(insurance$smoker) ## make smoker into a factor
summary(insurance$smoker) ## verify that origin is indeed a factor

library(GGally)
ggpairs(insurance[, 1:7]) ## produce a matrix of all bivariate plots


nzv <- nearZeroVar(ds.obj, saveMetrics= TRUE)
nzv

## next, let's split the data into two chunks - one used for training+testing; 
## the other to serve as a hold-out set that we use to assess model performance (scoring)
## we will use a simple splitting approach.
set.seed(1984)
trainIndices <- createDataPartition(ds.obj$charges, ## indicate which var. is outcome
                                    p = 0.8, # indicate proportion to use in training-testing
                                    list = FALSE, 
                                    times = 1)

training <- ds.obj[trainIndices,]
holdout <- ds.obj[-trainIndices,]

## centering and scaling as part of the pre-processing step
preProcValues <- preProcess(training, method = c("center", "scale"))

## Next, create the scaled+centered of the training+testing subset of the dataset
trainTransformed <- predict(preProcValues, training) 
## apply the same scaling and centering on the holdout set, too
holdoutTransformed <- predict(preProcValues, holdout)

## create settings for cross-validation to be used
## we will use repeated k-fold CV. For the sake of time
## we will use 5-fold CV with 3 repetitions
fitControl <- trainControl(
  method = "repeatedcv", ## perform repeated k-fold CV
  number = 5,
  repeats = 1)


##So we will vary the number of predictors in the range 1 to num. predictors in dataset

## And we'll create a random-forest model 
grid <- expand.grid(mtry = 1:(ncol(trainTransformed)-1)) 

forestfit <- train(charges ~ .,
                   data = trainTransformed, 
                   method = "rf",
                   trControl = fitControl,
                   verbose = FALSE,
                   tuneGrid = grid)


## check what information is available for the model fit
names(forestfit)

## some plots
trellis.par.set(caretTheme())
plot(forestfit)

## make predictions on the hold-out set
predvals <- predict(forestfit, holdoutTransformed)

# ## compute the performance metrics
postResample(pred = predvals, obs = holdoutTransformed$charges)

## Rank the variables in terms of their importance
varImp(forestfit)




## next, let's identify an optimal boosted-tree model

grid <- expand.grid(interaction.depth = seq(1:3),
                    shrinkage = seq(from = 0.01, to = 0.2, by = 0.01),
                    n.trees = seq(from = 100, to = 500, by = 100),
                    n.minobsinnode = seq(from = 5, to = 15, by = 5)
)

boostedfit <- train(charges ~ .,
                    data = trainTransformed, 
                    method = "gbm",
                    trControl = fitControl,
                    verbose = FALSE, ## setting this to TRUE or excluding this leads to a lot of output
                    tuneGrid = grid)

## check what information is available for the model fit
names(boostedfit)

## some plots
trellis.par.set(caretTheme())
plot(boostedfit)

## make predictions on the hold-out set
predvals <- predict(boostedfit, holdoutTransformed)
# 
# ## compute the performance metrics
postResample(pred = predvals, obs = holdoutTransformed$charges)

## Rank the variables in terms of their importance
varImp(boostedfit)




## next, let's identify an optimal bagged-tree model
grid <- expand.grid(mtry = ncol(trainTransformed)) 

forestfit <- train(charges ~ .,
                   data = trainTransformed, 
                   method = "rf",
                   trControl = fitControl,
                   verbose = FALSE,
                   tuneGrid = grid)


## check what information is available for the model fit
names(forestfit)

## some plots
trellis.par.set(caretTheme())
plot(forestfit)

## make predictions on the hold-out set
predvals <- predict(forestfit, holdoutTransformed)

# ## compute the performance metrics
postResample(pred = predvals, obs = holdoutTransformed$charges)

## Rank the variables in terms of their importance
varImp(forestfit)




## comparing the results of the randomforest algorithm with those from 
## the gbm algorithm, which one performed better? What about the values
## of variable importance - did the ranking stay the same or change?


stopCluster(cl)

