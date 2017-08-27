# PML Prediction Assignment
Mary T.  
August 28th, 2017  



## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Goal

In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and predict the manner in which they did the exercise, using the "classe" variable in the training set. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways: 
. exactly according to the specification (Class A),
. throwing the elbows to the front (Class B),
. lifting the dumbbell only halfway (Class C),
. lowering the dumbbell only halfway (Class D)
. throwing the hips to the front (Class E).

Other variables may be used for prediction. A report should be created describing how the model was built, how cross validation was used, what the expected out of sample error is, and why certain choices were made. The prediction model will also be used to predict 20 different test cases. 


## Data Preprocessing

Download and read the data.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.3.3
```

```r
library(rpart)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.3.3
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.3.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 3.3.3
```

```r
library(e1071)
```

```
## Warning: package 'e1071' was built under R version 3.3.3
```

```r
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./pml-training.csv"
testFile  <- "./pml-testing.csv"

download.file(trainUrl, destfile=trainFile, method="curl")
```

```
## Warning: running command 'curl "https://d396qusza40orc.cloudfront.net/
## predmachlearn/pml-training.csv" -o "./pml-training.csv"' had status 127
```

```
## Warning in download.file(trainUrl, destfile = trainFile, method = "curl"):
## download had nonzero exit status
```

```r
download.file(testUrl, destfile=testFile, method="curl")
```

```
## Warning: running command 'curl "https://d396qusza40orc.cloudfront.net/
## predmachlearn/pml-testing.csv" -o "./pml-testing.csv"' had status 127
```

```
## Warning in download.file(testUrl, destfile = testFile, method = "curl"):
## download had nonzero exit status
```

```r
trainRaw <- read.csv("./pml-training.csv")
testRaw <- read.csv("./pml-testing.csv")

dim(trainRaw)
```

```
## [1] 19622   160
```

```r
dim(testRaw)
```

```
## [1]  20 160
```
The training data set contains 19622 observations and 160 variables.
The testing data set contains 20 observations and 160 variables. 
The "classe" variable in the training set is the outcome to predict.

## Clean data

Remove columns with NA missing values. 
Remove columns that are meaningless to the prediction process.



```r
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 

classe <- trainRaw$classe
trainRemove <- grepl("^X|user_name|timestamp|window|^max|^min|^ampl|^var|^avg|^stdd|^ske|^kurt", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainNew <- trainRaw[, sapply(trainRaw, is.numeric)]
trainNew$classe <- classe
testRemove <- grepl("^X|user_name|timestamp|window|^max|^min|^ampl|^var|^avg|^stdd|^ske|^kurt", names(testRaw))
testRaw <- testRaw[, !testRemove]
testNew <- testRaw[, sapply(testRaw, is.numeric)]

dim(trainNew)
```

```
## [1] 19622    53
```

```r
dim(testNew)
```

```
## [1] 20 53
```
The new training data set now has 19622 observations and 53 variables.
The new testing data set contains 20 observations and 53 variables. 

## Data splitting

Split the new training set into a training data set (70%) and a validation data set (30%). 
The validation data set will be used to conduct cross-validation in future steps.


```r
set.seed(22519) 
inTrain <- createDataPartition(trainNew$classe, p=0.70, list=F)
trainData <- trainNew[inTrain, ]
testData <- trainNew[-inTrain, ]
```

## Data Modeling

Fit a predictive model using the Random Forest algorithm because it is able to classify large amounts of data with accuracy. Use the 5-fold cross validation when applying the algorithm.


```r
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10989, 10991, 10990, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9901727  0.9875673
##   27    0.9917015  0.9895017
##   52    0.9840572  0.9798282
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

Predict the performance of the model on the validation data set.


```r
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    1
##          B    5 1131    3    0    0
##          C    0    0 1021    5    0
##          D    0    0   13  949    2
##          E    0    0    1    6 1075
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9939          
##                  95% CI : (0.9915, 0.9957)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9923          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   1.0000   0.9836   0.9885   0.9972
## Specificity            0.9998   0.9983   0.9990   0.9970   0.9985
## Pos Pred Value         0.9994   0.9930   0.9951   0.9844   0.9935
## Neg Pred Value         0.9988   1.0000   0.9965   0.9978   0.9994
## Prevalence             0.2851   0.1922   0.1764   0.1631   0.1832
## Detection Rate         0.2843   0.1922   0.1735   0.1613   0.1827
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9984   0.9992   0.9913   0.9927   0.9979
```

```r
accuracy <- postResample(predictRf, testData$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9938828 0.9922620
```

```r
oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose
```

```
## [1] 0.006117247
```

The estimated accuracy of the model is 99.39%.
The estimated out-of-sample error is 0.61%.


## Predicting for Test Data Set

Apply the model to the original testing data (remove the problem_id column first).

```r
result <- predict(modelRf, testNew[, -length(names(testNew))])
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Generate prediction result "problem_id_x.txt" files for the 20 test cases.  


```r
pml_write_files <- function(x){

  n = length(x)

  for(i in 1:n){

    filename = paste0("problem_id_",i,".txt")

    write.table(x[i], file=filename, quote=FALSE,

                row.names=FALSE, col.names=FALSE)

  }

}

pml_write_files(result)
```


# ........................................................................................
# ........................................................................................
## Appendix

## Correlation Matrix


```r
corrMatrixPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrMatrixPlot, method="color")
```

![](PML_Project_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

## Decision Tree Visualization


```r
dtreeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(dtreeModel) 
```

![](PML_Project_files/figure-html/unnamed-chunk-9-1.png)<!-- -->
