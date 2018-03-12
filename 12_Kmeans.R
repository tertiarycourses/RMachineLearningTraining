#===================================================================
#                        Kmeans Clustering
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
ir2.train <- iris[inTrain,-5]
ir2.test <- iris[-inTrain,-5]
ir2.Class<- iris[-inTrain,5]
## clustering only deals with numeric data



### Making Tasks
kmeans.task = makeClusterTask(id ="ir2", data = ir2.train)
kmeans.task


########################## Making Learner

### Listing learners
lrns = listLearners()
head(lrns[c("class", "package")])


lrns = listLearners("cluster", properties = "prob")
head(lrns[c("class", "package")])

lrns$class  # see all our regression options


kmeans.lrn = makeLearner("cluster.kmeans", centers = 3)
# specify how many clusters centers you want


########################## Train the model
mod = train(kmeans.lrn, kmeans.task)
mod

names(mod)

### Extract the fitted model
getLearnerModel(mod)

######################## Predictions

kmeans.pred = predict(mod, newdata = ir2.test)
kmeans.pred

head(getPredictionResponse(kmeans.pred))


### visualize results
plotLearnerPrediction(kmeans.lrn, features=c("Petal.Length","Petal.Width"), 
                      task=kmeans.task)
