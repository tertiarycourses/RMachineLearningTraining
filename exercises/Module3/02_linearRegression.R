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
Boston=read.csv(file.choose())   # from MLM datasets
Boston=Boston[,-1]
View(Boston)

nr <- nrow(Boston)
inTrain <- sample(1:nr, 0.6*nr)
regr.train <- Boston[inTrain,]
regr.test <- Boston[-inTrain,]


### Making Tasks
regr.task = makeRegrTask(id = "bh", data = regr.train, 
                         target= "medv")
regr.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])

lrns = listLearners("classif") #, properties = "numerics")
head(lrns[c("class", "package")])

lrns$class  # see all our classification options


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
regr.pred = predict(mod, newdata = regr.test)
regr.pred

performance(regr.pred, measures = list(mlr::rmse, mlr::rsq))
plot(regr.pred$data$truth, regr.pred$data$response)

head(getPredictionTruth(regr.pred))

head(getPredictionResponse(regr.pred))


plot(getLearnerModel(mod, more.unwrap = TRUE))

### visualize results
plotLearnerPrediction(regr.lrn, features="lstat", task=regr.task)
