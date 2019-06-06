#===================================================================
#                        SVM Classifier
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
iris2=subset(iris, subset=iris$Species %in% c("versicolor","virginica"))
iris2$Species=factor(iris2$Species)


nr <- nrow(iris2)
inTrain <- sample(1:nr, 0.6*nr)
ir2.train <- iris2[inTrain,]
ir2.test <- iris2[-inTrain,]


### Making Tasks
svm.task = makeClassifTask(id = "ir2", 
                           data = ir2.train, 
                           target= "Species")
svm.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


svm.lrn = makeLearner("classif.svm")   #predict.type = "prob" (if you want probabililties)
svm.lrn

########################## Train the model
mod = train(svm.lrn, svm.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

svm.pred = predict(mod, newdata = ir2.test)
svm.pred

performance(svm.pred, measures = list(mmce, acc))

head(getPredictionTruth(svm.pred))

head(getPredictionResponse(svm.pred))

### Confusion Matrix

calculateConfusionMatrix(svm.pred)


### visualize results
plotLearnerPrediction(svm.lrn, features=c("Petal.Length",
                                          "Petal.Width"), 
                      task=svm.task)


fimp=generateFeatureImportanceData(task=svm.task,
                                   learner=svm.lrn)
ll=length(iris.test)-1
fimp$res
barplot(as.matrix(fimp$res[1:ll]), 
        names.arg =names(fimp$res[1:ll]),
        xlab=row.names(fimp$res),
        horiz=T)
