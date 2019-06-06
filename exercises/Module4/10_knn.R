#===================================================================
#                        KNN Classifier
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
knn.task = makeClassifTask(id ="iris", 
                           data = iris.train, 
                           target= "Species")
knn.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


knn.lrn = makeLearner("classif.knn")   


########################## Train the model
mod = train(knn.lrn, knn.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

knn.pred = predict(mod, newdata = iris.test)
knn.pred

performance(knn.pred, measures = list(mmce, acc))

head(getPredictionTruth(knn.pred))

head(getPredictionResponse(knn.pred))

### Confusion Matrix

calculateConfusionMatrix(knn.pred)


### visualize results
plotLearnerPrediction(knn.lrn, 
                      features=c("Petal.Length","Petal.Width"), 
                      task=knn.task)

fimp=generateFeatureImportanceData(task=knn.task,
                                   learner=knn.lrn)
ll=length(iris.test)-1
fimp$res
barplot(as.matrix(fimp$res[1:ll]), 
        names.arg =names(fimp$res[1:ll]),
        xlab=row.names(fimp$res),
        horiz=T)
