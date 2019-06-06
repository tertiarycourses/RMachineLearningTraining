#===================================================================
#                        Ridge Regression
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


glm.lrn = makeLearner("regr.glmnet")
glm.lrn

########################## Train the model
mod = train(glm.lrn, regr.task)
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


############ Tuning Regression models

# set limits
ps_glmnet <- makeParamSet(makeNumericParam("s", lower = 140, upper = 208))

# tune params in parallel using a grid search for simplicity
tune.ctrl = makeTuneControlGrid()
inner <- makeResampleDesc("CV", iters = 10)

msrs = list(mse, rsq)


lrn_glmnet <- makeLearner("regr.glmnet",
                          alpha = 0,                # rideg regression > alpha is zero
                          intercept = FALSE)


ps_glmnet2 = makeParamSet(
  makeDiscreteParam("alpha", values = 0),
  makeDiscreteParam("lambda", values = seq(0.001,0.2,length=5))
)

params_tuned_glmnet2 = tuneParams(lrn_glmnet, task = bh.task, resampling = inner,
                                  par.set = ps_glmnet2, control = tune.ctrl, 
                                  measure = msrs)

params_tuned_glmnet2$x
params_tuned_glmnet2$y


lrn_glmnet_ridge <- makeLearner("regr.glmnet",
                                alpha = 0,
                                lambda=0.001,
                                intercept = FALSE)


mod_ridge = mlr::train(lrn_glmnet_ridge, bh.task)



## Predictions
ridge.pred = predict(mod_ridge, newdata = regr.test)
ridge.pred

performance(ridge.pred, measures = list(mlr::rmse, mlr::rsq))
plot(ridge.pred$data$truth, ridge.pred$data$response)

head(getPredictionTruth(ridge.pred))

head(getPredictionResponse(ridge.pred))
