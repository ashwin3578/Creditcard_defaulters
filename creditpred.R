#PREDICTIVE ANALYTICS ASSIGNMENT (301117)

#STUDENT NO.: 19257701

#STUDENT NAME: ASHWIN MOHAN


#-------------------------------------------------------------------------------------------------

#QUESTION (1) DATA READING AND FORMATTING

#reading the csv into a
a = read.csv("Credit_Card_train.csv")

#converting to dataframe and storing into aframe
data.frame(a)
aframe=data.frame(a)

#-------------------------------------------------------------------------------------------------

#QUESTION (2) MISSING VALUE HANDLING

#seeing the entire AGE data from the dataframe aframe
aframe$AGE

#seeing the NAs in the AGE data 
aframe$AGE[is.na(aframe$AGE)]

#checking whether NAs are present anywhere else in the CSV apart from AGE
aframe[is.na(aframe)]
aframe$AGE[is.na(aframe$AGE)]
#It was found that the number of NAs in AGE is the same as the file, hence there are NAs only in AGE in the entire dataset

#Hence, we can use kNN Imputation. This replaces all the missing values in all the variables.
install.packages("DMwR")
library(DMwR)

#placing the NA replaced dataframe into bframe. Our new dataframe is bframe from here
bframe <- knnImputation(aframe, k=10, scale = TRUE,meth = "weighAvg", distData = NULL)

#verifying whether there are any NAs inside bframe
anyNA(bframe)
#No NAs were in the dataframe bframe

bframe

#Reference 1: Lecture 5 (Predictive Analytics)
#Reference 2: http://r-statistics.co/Missing-Value-Treatment-With-R.html
#Reference 3: https://www.rdocumentation.org/packages/bnstruct/versions/1.0.6/topics/knn.impute

#--------------------------------------------------------------------------------------------------

#QUESTION (3a) Determine what variables you want to use in the model 

#METHOD 1: BORUTA

#installing Boruta package to use the Boruta method to determine whether a variable is important or not
install.packages("Boruta")
library("Boruta")

#Performing a boruta search
boruta_output <- Boruta(default.payment.next.month ~ ., data=na.omit(bframe), doTrace=2)

boruta_output
# Boruta performed 51 iterations in 24.86463 mins.
# 21 attributes confirmed important: AGE, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4 and 16 more;
# 3 attributes confirmed unimportant: EDUCATION, ID, SEX;

#Store confirmed and tentative variables into boruta_signif
boruta_signif <- names(boruta_output$finalDecision[boruta_output$finalDecision %in% c("Confirmed", "Tentative")])

#Showing the important variables
boruta_signif
# [1] "LIMIT_BAL" "MARRIAGE"  "AGE"       "PAY_0"     "PAY_2"     "PAY_3"     "PAY_4"     "PAY_5"     "PAY_6"    
# [10] "BILL_AMT1" "BILL_AMT2" "BILL_AMT3" "BILL_AMT4" "BILL_AMT5" "BILL_AMT6" "PAY_AMT1"  "PAY_AMT2"  "PAY_AMT3" 
# [19] "PAY_AMT4"  "PAY_AMT5"  "PAY_AMT6" 
#The above are the important variables that I am using in my model. 
#The variables to be dropped are "ID", "SEX", "EDUCATION"

#Plotting variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")

# Do a tentative rough fix
roughFixMod <- TentativeRoughFix(boruta_output)
#It was found that there are no tentative attributes

#Reference 4: http://r-statistics.co/Variable-Selection-and-Importance-With-R.html

#---------------------------------------------------------------------------------------------------

#QUESTION (3b & 3c)

# Logistic regression, random forest, and SVM have been used
# Cross validation required in 3c has been shown along with the models

install.packages("caret")
library(caret)

#Splitting dataset into training and testing dataset in the ratio 70:30 using caret library
inTrain<- createDataPartition(y=bframe$default.payment.next.month, p=0.70, list=FALSE)
training<-bframe[inTrain,]
testing<-bframe[-inTrain,] 

dim(training)
dim(testing)
#The training dataset has 11046 rows while the testing set has 4733 rows. Both have 25 columns. 

training$default.payment.next.month <- as.factor(training$default.payment.next.month)
testing$default.payment.next.month <- as.factor(testing$default.payment.next.month)

install.packages("mlr")
library(mlr)

trainTask <- makeClassifTask(data = training, target = "default.payment.next.month")
testTask <- makeClassifTask(data = testing, target = "default.payment.next.month")

#Removing the variables that we discovered in the previous section were NOT IMPORTANT
trainTask <- dropFeatures(task = trainTask,features = c("ID","SEX","EDUCATION"))
testTask <- dropFeatures(task = testTask,features = c("ID","SEX","EDUCATION"))

#MODEL 1: LOGISTIC REGRESSION MODEL

logistic.learner <- makeLearner("classif.logreg",predict.type = "response")

#cross validation
cv.logistic <- crossval(learner = logistic.learner,task = trainTask,iters = 3,stratify = TRUE,measures = acc,show.info = F)

#cv accuracy
cv.logistic$aggr
# acc.test.mean 
# 0.8049973 

#training the model
fmodel <- train(logistic.learner,trainTask)
fmodel

#performing prediction on test data
fpmodel <- predict(fmodel, testTask)

fpmodel
# Prediction: 4733 observations
# predict.type: response
# threshold: 
#   time: 0.04
# id truth response
# 3   1     0        0
# 5   2     0        0
# 8   3     0        0
# 11  4     0        0
# 15  5     0        0
# 16  6     0        0
# ... (#rows: 4733, #cols: 3)

#determining accuracy of prediction on test data
result = table(fpmodel$data$response,testing$default.payment.next.month)
result
#     0    1
# 0 3552  798
# 1  125  258

accuracy_Test <- sum(diag(result)) / sum(result)
print(paste('Accuracy for test', accuracy_Test))
# [1] "Accuracy for test 0.799915487006127"

#MODEL 2: RANDOM FOREST

#Getting the parameter list for hypertuning
getParamSet("classif.randomForest")

#creating a learner
rf <- makeLearner("classif.randomForest", predict.type = "response", par.vals = list(ntree = 200, mtry = 3))
rf$par.vals <- list(
  importance = TRUE
)

rf_param <- makeParamSet(
makeIntegerParam("ntree",lower = 50, upper = 500),
makeIntegerParam("mtry", lower = 3, upper = 10),
makeIntegerParam("nodesize", lower = 10, upper = 50)
)

#Performing random search on 50 iterations
rancontrol <- makeTuneControlRandom(maxit = 50L)

#Set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#hypertuning parameters
rf_tune <- tuneParams(learner = rf, resampling = set_cv, task = trainTask, par.set = rf_param, control = rancontrol, measures = acc)

# [Tune] Started tuning learner classif.randomForest for parameter set:
#   Type len Def    Constr Req Tunable Trafo
# ntree    integer   -   - 50 to 500   -    TRUE     -
#   mtry     integer   -   -   3 to 10   -    TRUE     -
#   nodesize integer   -   -  10 to 50   -    TRUE     -
#   With control class: TuneControlRandom
# Imputation value: -0
# [Tune-x] 1: ntree=245; mtry=3; nodesize=26
# [Tune-y] 1: acc.test.mean=0.8174000; time: 0.7 min
# [Tune-x] 2: ntree=155; mtry=3; nodesize=49
# [Tune-y] 2: acc.test.mean=0.8169473; time: 0.3 min
# [Tune-x] 3: ntree=143; mtry=9; nodesize=28
# [Tune-y] 3: acc.test.mean=0.8157704; time: 0.3 min
# [Tune-x] 4: ntree=103; mtry=9; nodesize=28
# [Tune-y] 4: acc.test.mean=0.8156799; time: 0.2 min
# [Tune-x] 5: ntree=56; mtry=7; nodesize=42
# [Tune-y] 5: acc.test.mean=0.8154988; time: 0.1 min
# [Tune-x] 6: ntree=139; mtry=9; nodesize=47
# [Tune-y] 6: acc.test.mean=0.8168568; time: 0.3 min
# [Tune-x] 7: ntree=355; mtry=9; nodesize=17
# [Tune-y] 7: acc.test.mean=0.8138693; time: 0.7 min
# [Tune-x] 8: ntree=208; mtry=6; nodesize=41
# [Tune-y] 8: acc.test.mean=0.8184863; time: 0.4 min
# [Tune-x] 9: ntree=210; mtry=9; nodesize=20
# [Tune-y] 9: acc.test.mean=0.8139598; time: 0.4 min
# [Tune-x] 10: ntree=246; mtry=3; nodesize=14
# [Tune-y] 10: acc.test.mean=0.8141409; time: 0.5 min
# [Tune-x] 11: ntree=78; mtry=9; nodesize=46
# [Tune-y] 11: acc.test.mean=0.8172189; time: 0.1 min
# [Tune-x] 12: ntree=85; mtry=9; nodesize=26
# [Tune-y] 12: acc.test.mean=0.8153178; time: 0.2 min
# [Tune-x] 13: ntree=479; mtry=7; nodesize=35
# [Tune-y] 13: acc.test.mean=0.8172189; time: 0.9 min
# [Tune-x] 14: ntree=444; mtry=5; nodesize=19
# [Tune-y] 14: acc.test.mean=0.8154988; time: 0.8 min
# [Tune-x] 15: ntree=237; mtry=5; nodesize=39
# [Tune-y] 15: acc.test.mean=0.8170378; time: 0.4 min
# [Tune-x] 16: ntree=142; mtry=5; nodesize=33
# [Tune-y] 16: acc.test.mean=0.8176716; time: 0.2 min
# [Tune-x] 17: ntree=471; mtry=6; nodesize=33
# [Tune-y] 17: acc.test.mean=0.8174905; time: 0.9 min
# [Tune-x] 18: ntree=474; mtry=8; nodesize=23
# [Tune-y] 18: acc.test.mean=0.8157704; time: 0.9 min
# [Tune-x] 19: ntree=371; mtry=8; nodesize=45
# [Tune-y] 19: acc.test.mean=0.8158609; time: 0.7 min
# [Tune-x] 20: ntree=476; mtry=8; nodesize=29
# [Tune-y] 20: acc.test.mean=0.8144125; time: 1.1 min
# [Tune-x] 21: ntree=129; mtry=4; nodesize=37
# [Tune-y] 21: acc.test.mean=0.8181242; time: 0.3 min
# [Tune-x] 22: ntree=371; mtry=5; nodesize=10
# [Tune-y] 22: acc.test.mean=0.8141409; time: 0.8 min
# [Tune-x] 23: ntree=247; mtry=8; nodesize=49
# [Tune-y] 23: acc.test.mean=0.8161325; time: 0.5 min
# [Tune-x] 24: ntree=266; mtry=6; nodesize=43
# [Tune-y] 24: acc.test.mean=0.8172189; time: 0.5 min
# [Tune-x] 25: ntree=304; mtry=7; nodesize=26
# [Tune-y] 25: acc.test.mean=0.8162231; time: 0.8 min
# [Tune-x] 26: ntree=189; mtry=5; nodesize=34
# [Tune-y] 26: acc.test.mean=0.8170378; time: 0.4 min
# [Tune-x] 27: ntree=422; mtry=5; nodesize=44
# [Tune-y] 27: acc.test.mean=0.8174905; time: 0.8 min
# [Tune-x] 28: ntree=114; mtry=4; nodesize=12
# [Tune-y] 28: acc.test.mean=0.8142314; time: 0.3 min
# [Tune-x] 29: ntree=215; mtry=3; nodesize=17
# [Tune-y] 29: acc.test.mean=0.8138693; time: 0.5 min
# [Tune-x] 30: ntree=263; mtry=6; nodesize=28
# [Tune-y] 30: acc.test.mean=0.8163136; time: 0.6 min
# [Tune-x] 31: ntree=74; mtry=10; nodesize=31
# [Tune-y] 31: acc.test.mean=0.8135977; time: 0.2 min
# [Tune-x] 32: ntree=221; mtry=8; nodesize=46
# [Tune-y] 32: acc.test.mean=0.8173094; time: 0.5 min
# [Tune-x] 33: ntree=326; mtry=9; nodesize=14
# [Tune-y] 33: acc.test.mean=0.8142314; time: 1.0 min
# [Tune-x] 34: ntree=231; mtry=5; nodesize=17
# [Tune-y] 34: acc.test.mean=0.8157704; time: 0.7 min
# [Tune-x] 35: ntree=381; mtry=9; nodesize=43
# [Tune-y] 35: acc.test.mean=0.8162231; time: 1.1 min
# [Tune-x] 36: ntree=445; mtry=3; nodesize=44
# [Tune-y] 36: acc.test.mean=0.8174905; time: 1.0 min
# [Tune-x] 37: ntree=472; mtry=6; nodesize=11
# [Tune-y] 37: acc.test.mean=0.8151367; time: 1.1 min
# [Tune-x] 38: ntree=150; mtry=10; nodesize=19
# [Tune-y] 38: acc.test.mean=0.8129640; time: 0.4 min
# [Tune-x] 39: ntree=434; mtry=9; nodesize=44
# [Tune-y] 39: acc.test.mean=0.8157704; time: 1.1 min
# [Tune-x] 40: ntree=195; mtry=8; nodesize=31
# [Tune-y] 40: acc.test.mean=0.8152272; time: 0.4 min
# [Tune-x] 41: ntree=374; mtry=8; nodesize=11
# [Tune-y] 41: acc.test.mean=0.8143219; time: 0.9 min
# [Tune-x] 42: ntree=205; mtry=10; nodesize=12
# [Tune-y] 42: acc.test.mean=0.8142314; time: 0.4 min
# [Tune-x] 43: ntree=457; mtry=6; nodesize=22
# [Tune-y] 43: acc.test.mean=0.8158609; time: 1.0 min
# [Tune-x] 44: ntree=258; mtry=6; nodesize=38
# [Tune-y] 44: acc.test.mean=0.8168568; time: 0.5 min
# [Tune-x] 45: ntree=482; mtry=8; nodesize=31
# [Tune-y] 45: acc.test.mean=0.8163136; time: 1.0 min
# [Tune-x] 46: ntree=285; mtry=9; nodesize=10
# [Tune-y] 46: acc.test.mean=0.8133261; time: 0.7 min
# [Tune-x] 47: ntree=125; mtry=4; nodesize=48
# [Tune-y] 47: acc.test.mean=0.8175810; time: 0.2 min
# [Tune-x] 48: ntree=387; mtry=4; nodesize=24
# [Tune-y] 48: acc.test.mean=0.8163136; time: 0.7 min
# [Tune-x] 49: ntree=406; mtry=8; nodesize=15
# [Tune-y] 49: acc.test.mean=0.8138693; time: 1.0 min
# [Tune-x] 50: ntree=246; mtry=6; nodesize=33
# [Tune-y] 50: acc.test.mean=0.8166757; time: 0.5 min
# [Tune] Result: ntree=208; mtry=6; nodesize=41 : acc.test.mean=0.8184863

#checking accuracy
rf_tune$y

# acc.test.mean 
# 0.8184863 

#best parameters
rf_tune$x
# $ntree
# [1] 222
# 
# $mtry
# [1] 3
# 
# $nodesize
# [1] 44

#using hyperparameters for the modelling
rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)
rf.tree

#training model
rforest <- train(rf.tree, trainTask)
rforest
# Model for learner.id=classif.randomForest; learner.class=classif.randomForest
# Trained on: task.id = training; obs = 11046; features = 24
# Hyperparameters: importance=TRUE,ntree=222,mtry=3,nodesize=44

install.packages("rpart")
library(rpart)
getLearnerModel(rforest)

#making predictions
rfmodel <- predict(rforest,testTask)
rfmodel

result = table(rfmodel$data$response,testing$default.payment.next.month)
result

accuracy_Test <- sum(diag(result)) / sum(result)
print(paste('Accuracy for test', accuracy_Test))

# [1] "Accuracy for test 0.80857806887809"

# MODEL 3: SVM (Support Vector Machines)

install.packages("kernlab")
library(kernlab)

getParamSet("classif.ksvm")  
ksvm <- makeLearner("classif.ksvm", predict.type = "response")

pssvm <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)
ctrl <- makeTuneControlGrid()

res <- tuneParams(ksvm, task = trainTask, resampling = set_cv, par.set = pssvm, control = ctrl,measures = acc)

# [Tune] Started tuning learner classif.ksvm for parameter set:
#   Type len Def                   Constr Req Tunable Trafo
# C     discrete   -   - 0.00390625,0.0625,0.25,1   -    TRUE     -
#   sigma discrete   -   -   0.00390625,0.0625,1,16   -    TRUE     -
#   With control class: TuneControlGrid
# Imputation value: -0
# [Tune-x] 1: C=0.00390625; sigma=0.00390625
# [Tune-y] 1: acc.test.mean=0.7744885; time: 0.4 min
# [Tune-x] 2: C=0.0625; sigma=0.00390625
# [Tune-y] 2: acc.test.mean=0.7747601; time: 0.3 min
# [Tune-x] 3: C=0.25; sigma=0.00390625
# [Tune-y] 3: acc.test.mean=0.7846279; time: 0.3 min
# [Tune-x] 4: C=1; sigma=0.00390625
# [Tune-y] 4: acc.test.mean=0.8050878; time: 0.3 min
# [Tune-x] 5: C=0.00390625; sigma=0.0625
# [Tune-y] 5: acc.test.mean=0.7744885; time: 0.2 min
# [Tune-x] 6: C=0.0625; sigma=0.0625
# [Tune-y] 6: acc.test.mean=0.8015571; time: 0.3 min
# [Tune-x] 7: C=0.25; sigma=0.0625
# [Tune-y] 7: acc.test.mean=0.8121492; time: 0.3 min
# [Tune-x] 8: C=1; sigma=0.0625
# [Tune-y] 8: acc.test.mean=0.8140503; time: 0.3 min
# [Tune-x] 9: C=0.00390625; sigma=1
# [Tune-y] 9: acc.test.mean=0.7744885; time: 0.4 min
# [Tune-x] 10: C=0.0625; sigma=1
# [Tune-y] 10: acc.test.mean=0.7744885; time: 0.7 min
# [Tune-x] 11: C=0.25; sigma=1
# [Tune-y] 11: acc.test.mean=0.7832700; time: 1.0 min
# [Tune-x] 12: C=1; sigma=1
# [Tune-y] 12: acc.test.mean=0.7925946; time: 0.9 min
# [Tune-x] 13: C=0.00390625; sigma=16
# [Tune-y] 13: acc.test.mean=0.7744885; time: 0.7 min
# [Tune-x] 14: C=0.0625; sigma=16
# [Tune-y] 14: acc.test.mean=0.7744885; time: 0.8 min
# [Tune-x] 15: C=0.25; sigma=16
# [Tune-y] 15: acc.test.mean=0.7744885; time: 1.0 min
# [Tune-x] 16: C=1; sigma=16
# [Tune-y] 16: acc.test.mean=0.7751222; time: 1.1 min
# [Tune] Result: C=1; sigma=0.0625 : acc.test.mean=0.8140503

res$y
# acc.test.mean 
# 0.8145935 

#Setting model with best parameters
t.svm <- setHyperPars(ksvm, par.vals = res$x)

#training
par.svm <- train(ksvm, trainTask)

#Performing prediction on test
predict.svm <- predict(par.svm, testTask)

predict.svm

result = table(predict.svm$data$response,testing$default.payment.next.month)
result
#     0    1
# 0 3488  718
# 1  189  338

accuracy_Test <- sum(diag(result)) / sum(result)
print(paste('Accuracy for test', accuracy_Test))
# [1] "Accuracy for test 0.808366786393408"

# IT was determined that the accuracy was 79.991% for logistic regression model, 
# ,80.857% for random forest, 80.808% for SVM.
# Hence, random forest is the best model having the highest accuracy of 80.604%
# Reference 5: https://www.analyticsvidhya.com/blog/2016/08/practicing-machine-learning-techniques-in-r-with-mlr-package/
