library(mlr)

# capLargeValues: Convert large/infinite numeric values.
# createDummyFeatures: Generate dummy variables for factor features.
# joinClassLevels: Only for classification: Merge existing classes to new,larger classes.
# mergeSmallFactorLevels: Merge infrequent levels of factor features.
# normalizeFeatures: Normalize features by different methods, e.g., standardization or scaling to a certain range.
# removeConstantFeatures: Remove constant features.


# subsetTask: Remove observations and/or features from a Task.
# dropFeatures: Remove selected features from a task

data(iris)

## selecting rows and columns

iris2.1=subset(iris, select=c("Sepal.Length","Sepal.Width"))   # select only these 2 columns
iris2.2=iris[1:100, ]                                          # select the first 100 rows

## random sampling

# take a random sample of size 50 rows from a dataset iris 
# sample without replacement
myiris <- iris[sample(1:nrow(iris), 50,
  	replace=FALSE),]


## impute missing values

data(airquality)
aqr=airquality
summary(aqr)

imp = impute(aqr, classes = list(integer = imputeMean(), factor = imputeMode()), dummy.classes = "integer")
summary(imp$data)

## remove constant columns
iris2.3=removeConstantFeatures(iris)  # remove any constant column


## normalize columns
iris2.4 = normalizeFeatures(iris[,1:4], method = "range") # normalize the variables
summary(iris2.4)


## create train and test set

nr <- nrow(iris)
inTrain <- sample(1:nr, 0.6*nr)
iris.train <- iris[inTrain,]
iris.test <- iris[-inTrain,]



####### LISTINGS

mlr::listTaskTypes()

#If you would like a list of available learners, maybe only with certain properties
#or suitable for a certain learning Task use function listLearners.
View(mlr::listLearners())

mlr::listMeasures()
#available measures with certain properties or suitable
#for a certain learning Task use the function listMeasures.

