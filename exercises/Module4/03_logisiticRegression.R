#===================================================================
#                        Logistic Regression Classifier
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
# iris2$Species=as.numeric(iris2$Species)-1
iris2$Species=factor(iris2$Species)


nr <- nrow(iris2)
inTrain <- sample(1:nr, 0.6*nr)
ir2.train <- iris2[inTrain,]
ir2.test <- iris2[-inTrain,]

### Making Tasks
log.task = makeClassifTask(id = "ir2", data = ir2.train, target= "Species")
log.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


log.lrn = makeLearner("classif.logreg", predict.type = "prob") #(if you want probabililties)
log.lrn

########################## Train the model
mod = train(log.lrn, log.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

log.pred = predict(mod, newdata = ir2.test)
log.pred

performance(log.pred, measures = list(mmce, acc))

head(getPredictionTruth(log.pred))

head(getPredictionResponse(log.pred))

### Confusion Matrix

calculateConfusionMatrix(log.pred)


### ROC curve

## for ROC the prediction must be type "prob"
## so we run the model again , but include this setting


df = generateThreshVsPerfData(log.pred, measures = list(fpr, tpr,mmce))
plotROCCurves(df)


### visualize results
plotLearnerPrediction(log.lrn, features=c("Petal.Length","Petal.Width"), 
                      task=log.task)