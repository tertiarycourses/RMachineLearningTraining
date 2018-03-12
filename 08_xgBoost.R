#===================================================================
#                        XGboost Classifier
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
data(iris)

nr <- nrow(iris)
inTrain <- sample(1:nr, 0.6*nr)
iris.train <- iris[inTrain,]
iris.test <- iris[-inTrain,]


### Making Tasks
xg.task = makeClassifTask(id ="iris", data = iris.train, target= "Species")
xg.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


xg.lrn = makeLearner("classif.xgboost")   


########################## Train the model
mod = train(xg.lrn, xg.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

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
