library(mlr)
library(glmnet)


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



########### plot


configureMlr(on.par.without.desc = "quiet")

bm3 = benchmark(learners = list(
               makeLearner("regr.glmnet", id="glm1", alpha = 0, lambda = 0.001),
               makeLearner("regr.glmnet", id="glm2", alpha = 1, lambda = 0.001),
               makeLearner("regr.glmnet", id="glm3", alpha = 0.5, lambda = 0.001)
  ), tasks = bh.task, resamplings = inner, measures = msrs)

getBMRAggrPerformances(bm3, as.df = TRUE)

