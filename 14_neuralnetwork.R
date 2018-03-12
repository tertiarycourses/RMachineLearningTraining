#===================================================================
#                        Neural Net Classifier
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
ir2.train <- iris[inTrain,]
ir2.test <- iris[-inTrain,]


### Making Tasks
nn.task = makeClassifTask(id ="ir2", data = ir2.train, target= "Species")
nn.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


nn.lrn = makeLearner("classif.nnet")   


########################## Train the model
mod = train(nn.lrn, nn.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

nn.pred = predict(mod, newdata = iris.test)
nn.pred

performance(nn.pred, measures = list(mmce, acc))

head(getPredictionTruth(nn.pred))

head(getPredictionResponse(nn.pred))

### Confusion Matrix

calculateConfusionMatrix(nn.pred)


### visualize results
plotLearnerPrediction(nn.lrn, features=c("Petal.Length","Petal.Width"), 
                      task=nn.task)
