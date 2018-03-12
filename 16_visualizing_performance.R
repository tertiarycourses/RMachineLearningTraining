############################ Benchmark Experiments


##### Making Tasks

### Splitting data
data(iris)

nr <- nrow(iris)
inTrain <- sample(1:nr, 0.6*nr)
iris.train <- iris[inTrain,]
iris.test <- iris[-inTrain,]


### Making Tasks
my.task = makeClassifTask(id ="iris", data = iris.train, target= "Species")
my.task


## let us compare all the trees
lrns = list(makeLearner("classif.rpart"),
            makeLearner("classif.randomForest"),
            makeLearner("classif.gbm"),
            makeLearner("classif.xgboost"))

### Choose the resampling strategy
rdesc = makeResampleDesc("CV", iters = 5)

### Conduct the benchmark experiment
bmr = benchmark(lrns, my.task, rdesc)
bmr

## get the performance
getBMRPerformances(bmr)

getBMRAggrPerformances(bmr)

### get the predictions
getBMRPredictions(bmr)

### learner models
getBMRModels(bmr)

getBMRLearners(bmr)

getBMRMeasures(bmr)

## predictions
rin = getBMRPredictions(bmr)[[1]][[1]]$instance
rin


#### Visualizations

plotBMRBoxplots(bmr, measure = mmce)


plotBMRRanksAsBarChart(bmr)

# aggregated performance
plotBMRSummary(bmr)
