#===================================================================
#                        Decision Tree Classifier
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
rpart.task = makeClassifTask(id = "iris", 
                             data = iris.train, 
                             target= "Species")
rpart.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


rpart.lrn = makeLearner("classif.rpart",predict.type = "prob")  


########################## Train the model
mod = train(rpart.lrn, rpart.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

rpart.pred = predict(mod, newdata = iris.test)
rpart.pred

performance(rpart.pred, measures = list(mmce, acc))

head(getPredictionTruth(rpart.pred))

head(getPredictionResponse(rpart.pred))

### Confusion Matrix

calculateConfusionMatrix(rpart.pred)


### visualize results
plotLearnerPrediction(rpart.lrn, features=c("Petal.Length","Petal.Width"), 
                      task=rpart.task)
