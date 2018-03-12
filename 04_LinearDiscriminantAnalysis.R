#===================================================================
#                        Linear Discriminant Analysis
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
lda.task = makeClassifTask(id = "ir2", data = ir2.train, target= "Species")
lda.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


lda.lrn = makeLearner("classif.lda")   #predict.type = "prob" (if you want probabililties)
lda.lrn

########################## Train the model
mod = train(lda.lrn, lda.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

lda.pred = predict(mod, newdata = ir2.test)
lda.pred

performance(lda.pred, measures = list(mmce, acc))

head(getPredictionTruth(lda.pred))

head(getPredictionResponse(lda.pred))

### Confusion Matrix

calculateConfusionMatrix(lda.pred)


### visualize results
plotLearnerPrediction(lda.lrn, features=c("Petal.Length","Petal.Width"), 
                      task=lda.task)
