
#################### Resampling

# The supported resampling strategies are:
# Cross-validation ("CV"),
# Leave-one-out cross-validation ("LOO""),
# Repeated cross-validation ("RepCV"),
# Out-of-bag bootstrap and other variants ("Bootstrap"),
# Subsampling, also called Monte-Carlo cross-validaton ("Subsample"),
# Holdout (training/test) ("Holdout").
# The resample function evaluates the performance of a Learner using the specified
# resampling strategy for a given machine learning Task.


### Specify the resampling strategy (3-fold cross-validation)
rdesc = makeResampleDesc("CV", iters = 3)

## example
# we use our xgBoost example
xg.task
xg.lrn

r = resample("classif.xgboost", xg.task, rdesc)
r


################### Hyper-parameter tuning

#devtools::install_github("berndbischl/ParamHelpers", force=TRUE)
#devtools::install_github("jakob-r/mlrHyperopt", dep=T)

library(mlrHyperopt)
res = hyperopt(xg.task, learner = "classif.xgboost")    # this will take about 10 minutes
res

# try out out new results


data(iris)

nr <- nrow(iris)
inTrain <- sample(1:nr, 0.6*nr)
iris.train <- iris[inTrain,]
iris.test <- iris[-inTrain,]


xg.task = makeClassifTask(id ="iris", data = iris.train, target= "Species")
xg.task


xg.lrn = makeLearner("classif.xgboost", par.vals = list(nrounds = 663,    # using the results from above
                                                        max_depth = 10,
                                                        eta=0.571,
                                                        gamma=4.92,
                                                        colsample_bytree=0.686,
                                                        min_child_weight=3.16,
                                                        subsample=0.563))

mod = train(xg.lrn, xg.task)
mod

names(mod)


getLearnerModel(mod)

xg.pred = predict(mod, newdata = iris.test)
xg.pred

performance(xg.pred, measures = list(mmce, acc))

head(getPredictionTruth(xg.pred))

head(getPredictionResponse(xg.pred))

### Confusion Matrix

calculateConfusionMatrix(xg.pred)


### visualize results
plotLearnerPrediction(xg.lrn, features=c("Petal.Length","Petal.Width"), 
                      task=xg.task)
