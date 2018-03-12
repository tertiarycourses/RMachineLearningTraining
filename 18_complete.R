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


#======================================================================================================
#                                    PRE-PROCESSING DATA
#======================================================================================================

############################# Splitting Data

### Splitting data
data(iris)
iris2=subset(iris, subset=iris$Species %in% c("versicolor","virginica"))
iris2$Species=factor(iris2$Species)


nr <- nrow(iris2)
inTrain <- sample(1:nr, 0.6*nr)
ir2.train <- iris2[inTrain,]
ir2.test <- iris2[-inTrain,]


########################## making Recepie
library(recipes)

rec = recipe(Species ~ .,data = ir2.train) %>%
  step_corr(all_predictors(), threshold = .85) %>%
  step_nzv(all_predictors()) %>%                  # remove near zero variance variable
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  step_pca(all_predictors())

summary(rec)
formula(rec)

rec = prep(rec, training = ir2.train)
# If your training set doesn't pass, prep() will stop with an error

train = bake(rec, newdata = ir2.train)
test = bake(rec, newdata = ir2.test)


######## checking

# check_cols creates a specification of a recipe step that will check if all the columns of the training
# frame are present in the new data.

# check_missing creates a a specification of a recipe operation that will check if variables contain
# missing values.
  
######### remove variables  

#  step_corr attempts to remove variables to keep the largest absolute correlation between the variables
#  less than threshold.
  
#  step_rm creates a specification of a recipe step that will remove variables based on their name,
#  type, or role.
  
#  step_downsample creates a specification of a recipe step that will remove rows of a data set to
#  make the occurrence of levels in a specific factor level equal.

####### imputation

#  step_bagimpute creates a specification of a recipe step that will create bagged tree models to
#  impute missing data.

# there is also step_knnimpute/step_meanimpute/step_modeimpute

  
######### conversion  
  
#  step_bin2factor creates a specification of a recipe step that will create a two-level factor from a
#  single dummy variable.
  
#  step_num2factor will convert one or more numeric vectors to factors (ordered or unordered). This
#  can be useful when categories are encoded as integers.
  
#  step_ordinalscore creates a specification of a recipe step that will convert ordinal factor variables
#  into numeric scores.
  
#  step_classdist will create a
#  The function will create a new column for every unique value of the class variable.  
  
#  step_discretize creates a a specification of a recipe step that will convert numeric data into a
#  factor with bins having approximately the same number of data points (based on a training set).

#  step_dummy creates a a specification of a recipe step that will convert nominal data (e.g. character
#  or factors) into one or more numeric binary model terms for the levels of the original data.
  
#  step_factor2string will convert one or more factor vectors to strings.
#  step_string2factor will convert one or more character vectors to factors (ordered or unordered).
  
#  step_interact creates a specification of a recipe step that will create new columns that are interaction
#  terms between two or more variables.

########## Transformation  

#  step_spatialsign is a specification of a recipe step that will convert numeric data into a projection
#  on to a unit sphere.      
#  step_hyperbolic creates a specification of a recipe step that will transform data using a hyperbolic
#  function.  
#  step_invlogit creates a specification of a recipe step that will transform the data from real values
#  to be between zero and one.
#  step_log creates a specification of a recipe step that will log transform data.
#  step_logit creates a specification of a recipe step that will logit transform data.
#  step_sqrt creates a specification of a recipe step that will square root transform the data.
#  step_YeoJohnson creates a specification of a recipe step that will transform data using a simple
#  Yeo-Johnson transformation.
#  step_BoxCox creates a specification of a recipe step that will transform data using a simple Box-
#    Cox transformation.
  
       
#  add_role(sample, new_role = "id variable") %>%
#  add_role(dataset, new_role = "splitting indicator")
  

#   For convenience, there are also functions that are more specific: all_numeric(),
#   all_nominal(), all_predictors(), and all_outcomes().


#======================================================================================================
#                                   MAKING THE MODEL
#======================================================================================================

library(mlr)

### Making Tasks
log.task = makeClassifTask(id = "ir2", data = train, target= "Species")
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

log.pred = predict(mod, newdata = test)
log.pred



#====================================================================================================
#                                MODEL PERFORMANCE
#===================================================================================================

performance(log.pred, measures = list(mmce, acc))

head(getPredictionTruth(log.pred))

head(getPredictionResponse(log.pred))

### Confusion Matrix

calculateConfusionMatrix(log.pred)


### Feature Importance
fi=generateFeatureImportanceData(task =log.task, learner=log.lrn)
fi

### visualize results
plotLearnerPrediction(log.lrn, features=c("PC1","PC2"), 
                      task=log.task)

plotResiduals(log.pred)

#### more visualizations

cal=generateCalibrationData(log.pred)
plotCalibration(cal)


# mlr::plotBMRBoxplots()             #bmr
# mlr::plotBMRRanksAsBarChart()      #bmr
# mlr::plotBMRSummary()              #bmr
# mlr::plotCalibration()             #calibration data
# mlr::plotCritDifferences()         #calibration data
# mlr::plotFilterValuesGGVIS()       #filter values
# mlr::plotHyperParsEffect()         #hyperpars effect data
# mlr::plotLearningCurveGGVIS()      #calibraion data
# mlr::plotPartialDependenceGGVIS()  #calibration data
# mlr::plotROCCurves()               # thresholdVSpref data
# mlr::plotThreshVsPerfGGVIS()       #thresholdVSpref data
# mlr::plotTuneMultiCritResultGGVIS() #tuneParMultiCrit
# mlr::plotViperCharts()              # list of predictions/resample results             
# 
# 
# mlr::generateCalibrationData()        # list of predictions/resample results
# mlr::generateCritDifferencesData()    #bmr
# mlr::generateFeatureImportanceData()
# mlr::generateFilterValuesData()
# mlr::generateFunctionalANOVAData()
# mlr::generateHyperParsEffectData()    #tune result
# mlr::generateLearningCurveData()
# mlr::generatePartialDependenceData()
# mlr::generateThreshVsPerfData()


fv=generateFilterValuesData(log.task)
plotFilterValuesGGVIS(fv)

ano=generateFunctionalANOVAData(mod,test)   # only for regression

lcd=generateLearningCurveData(log.lrn,log.task)
plotLearningCurveGGVIS(lcd)

pdd=generatePartialDependenceData(mod,test)
plotPartialDependenceGGVIS(pdd)

tpd = generateThreshVsPerfData(log.pred, measures = list(fpr, tpr,mmce))    # only for binary
plotROCCurves(tpd)
plotThreshVsPerfGGVIS(tpd)


library(breakDown)

plot(broken(mod$learner.model, test[1,]))
plot(broken(mod$learner.model, test[1,], trans = function(x) exp(x)/(1+exp(x))))

