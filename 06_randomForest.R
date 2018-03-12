#===================================================================
#                        Random Forest Classifier
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
rf.task = makeClassifTask(id ="iris", data = iris.train, target= "Species")
rf.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


rf.lrn = makeLearner("classif.randomForest")   


########################## Train the model
mod = train(rf.lrn, rf.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

rf.pred = predict(mod, newdata = iris.test)
rf.pred

performance(rf.pred, measures = list(mmce, acc))

head(getPredictionTruth(rf.pred))

head(getPredictionResponse(rf.pred))

### Confusion Matrix

calculateConfusionMatrix(rf.pred)


### visualize results
plotLearnerPrediction(rf.lrn, features=c("Petal.Length","Petal.Width"), 
                      task=rf.task)
