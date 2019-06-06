devtools::install_github("laresbernardo/lares")
devtools::install_github("MI2DataLab/modelDown")
install.packages("mlr", dep=T)

library(mlr)

# https://github.com/tertiarycourses/RMachineLearning

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

imp = impute(aqr, classes = list(integer = imputeMean(), 
                                 factor = imputeMode()), 
                                  dummy.classes = "integer")
summary(imp$data)

### dummy variables

heart=read.csv(file.choose())

library(fastDummies)
heart2=dummy_cols(heart, select_columns = c("sex","chest_pain","fbs","restecg","exang"))
heart2$sex=NULL
heart2$chest_pain=NULL
heart2$fbs=NULL
heart2$restecg=NULL
heart2$exang=NULL

## remove constant columns
iris2.3=removeConstantFeatures(iris)  # remove any constant column


## normalize columns
# for the numeric columns
iris2.4 = normalizeFeatures(iris[,1:4], method = "range") # normalize the variables
summary(iris2.4)


## convert predictor to 0 and 1
# if you are doing classification

heart$disease=as.numeric(heart$disease)-1   # convert to 1 and 0

## create train and test set

nr <- nrow(iris)
inTrain <- sample(1:nr, 0.6*nr)
iris.train <- iris[inTrain,]
iris.test <- iris[-inTrain,]


######## challenge
# using the heart dataset > prepare the data for machine learning
# using all the techninques above.

h1=normalizeFeatures(heart[,c(1,4,5,8)], method = "range")
h2=dummy_cols(heart, select_columns = c("sex","chest_pain","fbs","restecg","exang"))
h21=h2[ ,11:ncol(h2)]
pred=as.numeric(heart$disease)-1
heart22=cbind(h1,h21,pred)

nr <- nrow(heart22)
inTrain <- sample(1:nr, 0.6*nr)
heart22.train <- heart22[inTrain,]
heart22.test <- heart22[-inTrain,]

####### LISTINGS

mlr::listTaskTypes()

#If you would like a list of available learners, maybe only with certain properties
#or suitable for a certain learning Task use function listLearners.
View(mlr::listLearners())

mlr::listMeasures()
#available measures with certain properties or suitable
#for a certain learning Task use the function listMeasures.
View(mlr::listMeasures())
