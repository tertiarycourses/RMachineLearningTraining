#===================================================================
#                        Multiple Linear Regression
#===================================================================

library(mlr)

#The tasks are organized in a hierarchy, with the generic Task at the top. The
#following tasks can be instantiated and all inherit from the virtual superclass
#Task:
#  . RegrTask for regression problems,
# . ClassifTask for binary and multi-class classification problems 
# (cost-sensitive classification with class-dependent costs can be handled as well),
# . SurvTask for survival analysis,
# . ClusterTask for cluster analysis,
# . MultilabelTask for multilabel classification problems,
# . CostSensTask for general cost-sensitive classification (with example-specific costs).


############################# Making Tasks

### Splitting data
data(Boston)
nr <- nrow(Boston)
inTrain <- sample(1:nr, 0.6*nr)
bh.train <- Boston[inTrain,]
bh.test <- Boston[-inTrain,]


### Making Tasks
regr.task = makeRegrTask(id = "bh", data = bh.train, target= "medv")
regr.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("regr", properties = "numerics")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


regr.lrn = makeLearner("regr.lm")
regr.lrn

########################## Train the model
mod = train(regr.lrn, regr.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

bh.pred = predict(mod, newdata = bh.test)
bh.pred

performance(bh.pred, measures = list(rmse))

head(getPredictionTruth(bh.pred))

head(getPredictionResponse(bh.pred))

### visualize results
plotLearnerPrediction(regr.lrn, features="lstat", task=bh.task)



