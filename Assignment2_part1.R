######################################################################################################
# Name : Nadeesha Perera
# Date : 02/25/2018
# Topic : Applied Machine Learning - Assignment 2 - part 1
# Purpose : Implement SVM, Decision trees, Boosting to classify data of online news popularity data set
#####################################################################################################

# Reference
# https://www.analyticsvidhya.com/blog/2016/08/practicing-machine-learning-techniques-in-r-with-mlr-package/
# https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
# https://mlr-org.github.io/mlr-tutorial/devel/html/advanced_tune/index.html
# http://xgboost.readthedocs.io/en/latest/parameter.html
# https://cran.r-project.org/web/packages/rpart/rpart.pdf
# https://cran.r-project.org/web/packages/mlr/mlr.pdf
# https://www.youtube.com/watch?v=rzjkT1uLNi4
# https://github.com/zmjones/imc
# https://github.com/mlr-org/mlr-tutorial/blob/gh-pages/2.11/mlr-tutorial.pdf
# http://datamining.togaware.com/survivor/Complexity_cp.html



# Clear work environment
rm(list = ls(all = TRUE))

# Load required libraries
library(data.table)
library(mlr)
library(kernlab)
library(rpart)
library(xgboost)
library(ggplot2)
library(parallelMap)

# Starting parallelization in mode=socket with cpus=2.
parallelStartSocket(4)


# Load data:
context1 <- read.csv('OnlineNewsPopularity.csv', header = TRUE)

# Remove non predictor variables
context2 <- context1[,3:61]

############################################################################################
# Convert categorical variables to factors
###############################################################################################

str(context2)
summary(context2)

#########################################################################################
# Look at correlation matrix
cor_mat <- cor(context2, use = "complete.obs", method = "pearson")

# The following are highly correlated
# n_unique_tokens || n_non_stop_unique_tokens || n_non_stop_words || average_token_length || global_subjectivity
# kw_max_max || kw_min_min || 
# kw_avg_avg || kw_max_avg
# data_channel_is_bus || LDA_00
# data_channel_is_world || LDA_02
# data_channel_is_tech || LDA_04
# data_channel_is_entertainment || LDA_01
# self_reference_avg_sharess || self_reference_min_shares || self_reference_max_shares
# is_weekend || weekday_is_saturday || weekday_is_sunday
# global_sentiment_polarity || rate_positive_words || rate_negative_words
# global_rate_positive_words || rate_positive_words
# global_rate_negative_words || rate_negative_words
# global_subjectivity || avg_positive_polarity || max_positive_polarity
# avg_negative_polarity || min_negative_polarity || max_negative_polarity
# title_subjectivity || abs_title_sentiment_polarity
# 


context2$data_channel_is_lifestyle <- as.factor(context2$data_channel_is_lifestyle)
context2$data_channel_is_bus <- as.factor(context2$data_channel_is_bus)
context2$data_channel_is_entertainment <- as.factor(context2$data_channel_is_entertainment)
context2$data_channel_is_socmed <- as.factor(context2$data_channel_is_socmed)
context2$data_channel_is_tech <- as.factor(context2$data_channel_is_tech)
context2$data_channel_is_world <- as.factor(context2$data_channel_is_world)
context2$weekday_is_monday <- as.factor(context2$weekday_is_monday)
context2$weekday_is_tuesday <- as.factor(context2$weekday_is_tuesday)
context2$weekday_is_wednesday <- as.factor(context2$weekday_is_wednesday)
context2$weekday_is_thursday <- as.factor(context2$weekday_is_thursday)
context2$weekday_is_friday <- as.factor(context2$weekday_is_friday)
context2$weekday_is_saturday <- as.factor(context2$weekday_is_saturday)
context2$weekday_is_sunday <- as.factor(context2$weekday_is_sunday)
context2$is_weekend <- as.factor(context2$is_weekend)


# Define train sample size
train_size <- floor(0.7*nrow(context2))

# Set seed to make sample reproducible
set.seed(4210)
train_ind <- sample(seq_len(nrow(context2)), size = train_size)

# Divide data to train and test
Train <- context2[train_ind,]
test <- context2[-train_ind,]


###################################################################################################
# Split the number of shares as high and low
################################################################################################


# Look at train data in shares to determine the boundary to seperate large and small shares
summary(Train$shares)
# The median is in 1,400. Will be a good place to split data

y_select <- Train[which(Train$shares < 10000),'shares']

histinfo <- hist(y_select, col = "blue", breaks = 100, xlab = "Number of shares", main = "Histogram of number of shares")
#histinfo

# y_log <- log(y_select)
# histinfo <- hist(y_log, col = "blue", breaks = 200, xlab = "Number of shares", main = "Histogram of number of shares")

# splitting data as large and small based on median

# Train data
y_train <- ifelse(Train$shares <= median(Train$shares), 0, 1)

# Final train set
#train_set_x <- as.matrix(Train[,1:58])
#train_set_y <- as.matrix(y_train)



# Final test set
train_set <- Train[,1:58]
train_set$shares_high <- as.factor(y_train)

# train_set_mat <- as.matrix(train_set) 

####################################################
# Split the test data
#####################################################

y_test <- ifelse(test$shares <= median(Train$shares), 0, 1)

# Final test set
test_set <- test[,1:58]
test_set$shares_high <- as.factor(y_test)

###########################################################################################################
# Explore the features
##############################################################################################################



qplot(train_set$n_tokens_title, train_set$n_tokens_content, data = train_set, color = train_set$shares_high)
qplot(train_set$n_tokens_title, train_set$n_unique_tokens, data = train_set, color = train_set$shares_high)
qplot(train_set$num_imgs, train_set$average_token_length, data = train_set, color = train_set$shares_high)
qplot(train_set$num_hrefs, train_set$num_imgs, data = train_set, color = train_set$shares_high)
qplot(train_set$num_hrefs, train_set$num_keywords, data = train_set, color = train_set$shares_high)
qplot(train_set$num_keywords, train_set$data_channel_is_entertainment, data = train_set, color = train_set$shares_high)

# plot(train_set[,16:20])
hist(train_set$n_tokens_content)
plot(train_set$n_tokens_content, train_set$shares_high, col = "blue", main = "n_tokens_content against shares", xlab = "n_tokens_content", ylab = "shares")
# There are few data points in this range. The distribution is right skewed. So let be




# check out summary
summary(train_set)
# Can't detect any anomalies.
# Since it is converted to binary - no outliers in the shares variable.
# no missing data or NA values
# The variable ranges are very different. Need to normalize



############################################################################################
# Feature selection
###############################################################################################

# Based on correlation finding select features to avoid multicollinearity
# ("n_tokens_title", "n_tokens_content", "n_unique_tokens", "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos", "num_keywords", "data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus", "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world", "kw_min_min", "kw_max_min", "kw_min_max", "kw_min_avg", "kw_max_avg", "self_reference_avg_sharess", "is_weekend", "LDA_03", "global_subjectivity", "global_sentiment_polarity", "avg_negative_polarity", "title_subjectivity", "title_sentiment_polarity", "shares_high")
features <- c(1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 24, 25, 29, 37, 41, 43, 44, 52, 55, 56, 59)

train_select <- train_set[,features]
test_select <- test_set[,features]

str(train_select)

############################################################################################
# Create a task
###############################################################################################

train_task <- makeClassifTask(data = train_select, target = "shares_high")
test_task <- makeClassifTask(data = test_select, target = "shares_high")


#normalize the variables
train_task <- normalizeFeatures(train_task,method = "standardize", range = c(0,1))
test_task <- normalizeFeatures(test_task,method = "standardize", range = c(0,1))



#################################################################################################################
#
# SVM
#
#######################################################################################################

getParamSet("classif.ksvm")

#set 5 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 5L)


####################################################################################################
# Linear Kernel
####################################################################################################

ksvm1 <- makeLearner("classif.ksvm", kernel = "vanilladot", predict.type = "response")

#Set parameters for rbfdot kernel
pssvm1 <- makeParamSet(
  makeDiscreteParam("C", values = c(0.01, 0.1, 1.0, 10))
  #makeDiscreteParam("kernel", values = c("rbfdot", "vanilladot" )),
  #makeDiscreteParam("sigma", values = c(0.0001, 0.001, 0.01, 0.1, 1.0))
  #                 requires = quote(kernel == "rbfdot")),
  #makeIntegerParam("degree", lower = 2L, upper = 5L,
  #                 requires = quote(kernel == "polydot"))
)

print(pssvm1)
#specify search function
ctrl1 <- makeTuneControlGrid()



#tune model
res1 <- tuneParams(ksvm1, task = train_task, resampling = set_cv, par.set = pssvm1, control = ctrl1, measures = list(acc, mmce, fpr, tpr, timetrain, timepredict))

# inspect all points evaluated during the search
data1 = generateHyperParsEffectData(res1)

# the performance over iterations
plotHyperParsEffect(data1, x = "iteration", y = "acc.test.mean", z = "C",  plot.type = "line")
plotHyperParsEffect(data1, x = "timetrain.test.mean", y = "acc.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data1, x = "C", y = "acc.test.mean", plot.type = "line")
plotHyperParsEffect(data1, x = "timepredict.test.mean", y = "acc.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data1, x = "exec.time", y = "acc.test.mean", plot.type = "line")
plotHyperParsEffect(data1, x = "iteration", y = "tpr.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data1, x = "iteration", y = "fpr.test.mean", z = "C", plot.type = "line")


#CV accuracy
res1$y

# The best parameters
res1$x


#set the model with best params
t.svm1 <- setHyperPars(ksvm1, par.vals = res1$x)

#train
par.svm1 <- train(ksvm1, train_task)

#test
predict.svm1 <- predict(par.svm1, test_task)

#submission file
submit1 <- data.frame(Actual_status = test_select$shares_high, Predict_Status = predict.svm1$data$response)

table(submit1)









####################################################################################################
# Gaussian Kernel
####################################################################################################

ksvm <- makeLearner("classif.ksvm", predict.type = "response")

#Set parameters for rbfdot kernel
pssvm <- makeParamSet(
  makeDiscreteParam("C", values = c(0.01, 0.1, 1.0, 10, 100)),
  #makeDiscreteParam("kernel", values = c("rbfdot", "vanilladot" )),
  makeDiscreteParam("sigma", values = c(0.0001, 0.001, 0.01, 0.1, 1.0))
  #                 requires = quote(kernel == "rbfdot")),
  #makeIntegerParam("degree", lower = 2L, upper = 5L,
  #                 requires = quote(kernel == "polydot"))
)

print(pssvm)
#specify search function
ctrl <- makeTuneControlGrid()


#tune model
res <- tuneParams(ksvm, task = train_task, resampling = set_cv, par.set = pssvm, control = ctrl, measures = list(acc, mmce, fpr, tpr, timetrain, timepredict))

# inspect all points evaluated during the search
data = generateHyperParsEffectData(res)

# the performance over iterations
plotHyperParsEffect(data, x = "iteration", y = "acc.test.mean", z = "C",  plot.type = "line")
plotHyperParsEffect(data, x = "timetrain.test.mean", y = "acc.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data, x = "C", y = "acc.test.mean", plot.type = "line")
plotHyperParsEffect(data, x = "sigma", y = "acc.test.mean", plot.type = "line")
plotHyperParsEffect(data, x = "timepredict.test.mean", y = "acc.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data, x = "exec.time", y = "acc.test.mean", plot.type = "line")
plotHyperParsEffect(data, x = "iteration", y = "tpr.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data, x = "iteration", y = "fpr.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data, x = "iteration", y = "fpr.test.mean", z = "sigma", plot.type = "line")


#CV accuracy
res$y

# The best parameters
res$x


#set the model with best params
t.svm <- setHyperPars(ksvm, par.vals = res$x)

#train
par.svm <- train(ksvm, train_task)

#test
predict.svm <- predict(par.svm, test_task)

#submission file
submit <- data.frame(Actual_status = test_select$shares_high, Predict_Status = predict.svm$data$response)

table(submit)





####################################################################################################
# Sigmoid Kernel
####################################################################################################

ksvm3 <- makeLearner("classif.ksvm", kernel = "tanhdot", predict.type = "response")


#Set parameters for rbfdot kernel
pssvm3 <- makeParamSet(
  makeDiscreteParam("C", values = c(0.001, 0.01, 0.1, 1.0, 10))
)

print(pssvm3)
#specify search function
ctrl3 <- makeTuneControlGrid()

#tune model
res3 <- tuneParams(ksvm3, task = train_task, resampling = set_cv, par.set = pssvm3, control = ctrl3, measures = list(acc, mmce, fpr, tpr, timetrain, timepredict))

# inspect all points evaluated during the search
data3 = generateHyperParsEffectData(res3)

# the performance over iterations
plotHyperParsEffect(data3, x = "iteration", y = "acc.test.mean", z = "C",  plot.type = "line")
plotHyperParsEffect(data3, x = "timetrain.test.mean", y = "acc.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data3, x = "C", y = "acc.test.mean", plot.type = "line")
plotHyperParsEffect(data3, x = "timepredict.test.mean", y = "acc.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data3, x = "exec.time", y = "acc.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data3, x = "iteration", y = "tpr.test.mean", z = "C", plot.type = "line")
plotHyperParsEffect(data3, x = "iteration", y = "fpr.test.mean", z = "C", plot.type = "line")


#CV accuracy
res3$y

# The best parameters
res3$x


#set the model with best params
t.svm3 <- setHyperPars(ksvm3, par.vals = res3$x)

#train
par.svm3 <- train(ksvm3, train_task)

#test
predict.svm3 <- predict(par.svm3, test_task)

#submission file
submit3 <- data.frame(Actual_status = test_select$shares_high, Predict_Status = predict.svm3$data$response)

table(submit3)





####################################################################################################
# Decision tree
####################################################################################################

getParamSet("classif.rpart")

#make tree learner
makeatree <- makeLearner("classif.rpart", predict.type = "response")


#Search for hyperparameters
gs <- makeParamSet(
  #makeIntegerParam("minsplit",lower = 10, upper = 1000),
  #makeIntegerParam("minbucket", lower = 5, upper = 1000),
  makeDiscreteParam("cp", values = c(0, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2))
)

#do a grid search
gscontrol <- makeTuneControlGrid() 

#hypertune the parameters
stune <- tuneParams(learner = makeatree, resampling = set_cv, task = train_task, par.set = gs, control = gscontrol, measures = list(acc, mmce, fpr, tpr, timetrain, timepredict))

# inspect all points evaluated during the search
data_tree = generateHyperParsEffectData(stune)

# the performance over iterations
plotHyperParsEffect(data_tree, x = "iteration", y = "acc.test.mean", z = "cp",  plot.type = "line")
plotHyperParsEffect(data_tree, x = "timetrain.test.mean", y = "acc.test.mean", z = "cp", plot.type = "line")
plotHyperParsEffect(data_tree, x = "cp", y = "acc.test.mean", plot.type = "line")
plotHyperParsEffect(data_tree, x = "timepredict.test.mean", y = "acc.test.mean", z = "cp", plot.type = "line")
plotHyperParsEffect(data_tree, x = "exec.time", y = "acc.test.mean", z = "cp", plot.type = "line")
plotHyperParsEffect(data_tree, x = "iteration", y = "tpr.test.mean", z = "cp", plot.type = "line")
plotHyperParsEffect(data_tree, x = "iteration", y = "fpr.test.mean", z = "cp", plot.type = "line")

qplot(data_tree$data$iteration, data_tree$data$tpr.test.mean, data = data_tree$data, color = data_tree$data$cp, geom = "path", xlab = "iteration", ylab = "tpr")
qplot(data_tree$data$iteration, data_tree$data$fpr.test.mean, data = data_tree$data, color = data_tree$data$cp, geom = "path", xlab = "iteration", ylab = "fpr")

# Check best parameter
stune$x

#cross validation result
stune$y

#using hyperparameters for modeling
t.tree <- setHyperPars(makeatree, par.vals = stune$x)

#train the model
t.rpart <- train(t.tree, train_task)
getLearnerModel(t.rpart)

#make predictions
tpmodel <- predict(t.rpart, test_task)

#create a submission file
submit_tree <- data.frame(Actual_status = test_select$shares_high, Predict_Status = tpmodel$data$response)

table(submit_tree)


####################################################################################################
# Boosted version of decision tree
####################################################################################################



# Remove non predictor variables
context3 <- context1[,3:61]


# Divide data to train and test
Train_boost <- context3[train_ind,]
test_boost <- context3[-train_ind,]


###################################################################################################
# Split the number of shares as high and low
################################################################################################


# Train data
y_train_boost <- ifelse(Train_boost$shares <= median(Train_boost$shares), 0, 1)


# Final test set
train_set_boost <- Train_boost[,1:58]
train_set_boost$shares_high <- as.factor(y_train_boost)

####################################################
# Split the test data
#####################################################

y_test_boost <- ifelse(test_boost$shares <= median(Train_boost$shares), 0, 1)

# Final test set
test_set_boost <- test_boost[,1:58]
test_set_boost$shares_high <- as.factor(y_test_boost)



# check out summary
summary(train_set_boost)
str(train_set_boost)
# Can't detect any anomalies.
# Since it is converted to binary - no outliers in the shares variable.
# no missing data or NA values
# The variable ranges are very different. Need to normalize



############################################################################################
# Feature selection
###############################################################################################

train_select_boost <- train_set_boost[,features]
test_select_boost <- test_set_boost[,features]

str(train_select_boost)

############################################################################################
# Create a task
###############################################################################################

train_task_boost <- makeClassifTask(data = train_select_boost, target = "shares_high")
test_task_boost <- makeClassifTask(data = test_select_boost, target = "shares_high")


#normalize the variables
train_task_boost <- normalizeFeatures(train_task_boost,method = "standardize", range = c(0,1))
test_task_boost <- normalizeFeatures(test_task_boost,method = "standardize", range = c(0,1))


############################################################################################
# Start boost algorithm
###############################################################################################





#load xgboost
set.seed(1001)
getParamSet("classif.xgboost")

#make learner with inital parameters
xg_set <- makeLearner("classif.xgboost", predict.type = "response")
xg_set$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 200
)

#define parameters for tuning
xg_ps <- makeParamSet(
  # The number of trees in the model (each one built sequentially)
  # makeIntegerParam("nrounds",lower=200,upper=600),
  # number of splits in each tree
  # makeIntegerParam("max_depth",lower=3,upper=20),
  # L2 regularization - prevents overfitting
  # makeNumericParam("lambda",lower=0.1,upper=1),
  # "shrinkage" - learning rate - prevents overfitting
  makeDiscreteParam("eta", values = c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1))
  # Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting
  # makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  # If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning
  # makeNumericParam("min_child_weight",lower=1,upper=5),
  # subsample ratio of columns when constructing each tree.
  # makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)

#define search function
rancontrol <- makeTuneControlGrid() 

#tune parameters
xg_tune <- tuneParams(learner = xg_set, task = train_task_boost, resampling = set_cv,measures = list(acc, mmce, fpr, tpr, timetrain, timepredict),par.set = xg_ps, control = rancontrol)


# inspect all points evaluated during the search
data_xg = generateHyperParsEffectData(xg_tune)

# the performance over iterations
plotHyperParsEffect(data_xg, x = "iteration", y = "acc.test.mean", z = "eta",  plot.type = "line")
plotHyperParsEffect(data_xg, x = "timetrain.test.mean", y = "acc.test.mean", z = "eta", plot.type = "line")
plotHyperParsEffect(data_xg, x = "eta", y = "acc.test.mean", plot.type = "line")
plotHyperParsEffect(data_xg, x = "timepredict.test.mean", y = "acc.test.mean", z = "eta", plot.type = "line")
plotHyperParsEffect(data_xg, x = "exec.time", y = "acc.test.mean", z = "eta", plot.type = "line")
plotHyperParsEffect(data_xg, x = "iteration", y = "tpr.test.mean", z = "eta", plot.type = "line")
plotHyperParsEffect(data_xg, x = "iteration", y = "fpr.test.mean", z = "eta", plot.type = "line")
plotHyperParsEffect(data_xg, x = "acc.test.mean", y = "tpr.test.mean", z = "eta", plot.type = "line")
plotHyperParsEffect(data_xg, x = "acc.test.mean", y = "fpr.test.mean", z = "eta", plot.type = "line")

qplot(data_xg$data$iteration, data_xg$data$tpr.test.mean, data = data_xg$data, color = data_xg$data$eta, geom = "path", xlab = "iteration", ylab = "tpr")
qplot(data_xg$data$iteration, data_xg$data$fpr.test.mean, data = data_xg$data, color = data_xg$data$eta, geom = "path", xlab = "iteration", ylab = "fpr")

xg_tune$x

xg_tune$y

#set parameters
xg_new <- setHyperPars(learner = xg_set, par.vals = xg_tune$x)

#train model
xgmodel <- train(xg_new, train_task_boost)

#test model
predict.xg <- predict(xgmodel, test_task_boost)

#submission file
submit_xg <- data.frame(Actual_status = test_select_boost$shares_high, Predict_Status = predict.xg$data$response)

table(submit_xg)







# Stopped parallelization. All cleaned up.
parallelStop()




