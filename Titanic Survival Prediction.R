#Title: Titanic Survival Prediction

#1. Load and Clean the Data
getwd()
setwd("C:/Users/user/Downloads")
Titanic_data <- read.csv("Titanic_dataset.csv")
View(Titanic_data)
# Load necessary libraries
library(dplyr)
library(ggplot2)
library(caret)

#2 Check the structure and summary of the dataset
colnames(Titanic_data)
# Check the structure of the dataset
str(Titanic_data)
# Summary of the dataset
summary(Titanic_data)

#3 Data Cleaning
#check missing value
any(is.na(Titanic_data))
#Comment: TRUE indicate missing value is present in the dataset.
# Check for missing values
colSums(is.na(Titanic_data))
#comment:
#1.PassengerId (0 missing): No missing values. This column is likely a unique identifier for each passenger.
#2.Survived (0 missing): No missing values. This column indicates whether the passenger survived.
#3.Pclass (0 missing): No missing values. This column represents the passenger's class (1st, 2nd, or 3rd).
#4.Name (0 missing): No missing values. This column contains the names of the passengers.
#5.Sex (0 missing): No missing values. This column indicates the gender of the passengers.
#6.Age (177 missing): There are 177 missing values in the Age column. This is a significant number of missing values and needs to be addressed because age is an important feature for predicting survival.
#7.SibSp (0 missing): No missing values. This column indicates the number of siblings or spouses aboard the Titanic.
#8.Parch (0 missing): No missing values. This column indicates the number of parents or children aboard the Titanic.
#9.Ticket (0 missing): No missing values. This column contains ticket numbers.
#10.Fare (0 missing): No missing values. This column represents the fare paid by the passenger.
#11.Cabin (687 missing): There are 687 missing values in the Cabin column, which is the majority of the data. Due to the high number of missing values, this column is often dropped from analysis.
#12.Embarked (0 missing): No missing values. This column indicates the port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

# Fill missing Age values with the median age
Titanic_data$Age[is.na(Titanic_data$Age)] <- median(Titanic_data$Age, na.rm = TRUE)

# Fill missing Embarked values with the mode
Titanic_data$Embarked[is.na(Titanic_data$Embarked)] <- as.character(stats::setNames(sort(table(Titanic_data$Embarked), decreasing = TRUE), NULL)[1])

# Drop the Cabin column due to too many missing values
Titanic_data <- Titanic_data %>% select(-Cabin)

# Convert categorical variables to factors
Titanic_data$Survived <- as.factor(Titanic_data$Survived)
Titanic_data$Pclass <- as.factor(Titanic_data$Pclass)
Titanic_data$Sex <- as.factor(Titanic_data$Sex)
Titanic_data$Embarked <- as.factor(Titanic_data$Embarked)
 
#again check missing_value
any(is.na(Titanic_data))
#comment: FALSE indicate there is no missing value in the dataset.

#4 Split the Data into Training and Testing Sets
# Set a seed for fixing
set.seed(123)
# Split the data
trainIndex <- createDataPartition(Titanic_data$Survived, p = 0.7, list = FALSE, times = 1)
trainData <- Titanic_data[trainIndex, ]
testData <- Titanic_data[-trainIndex, ]

#5  Build a Predictive Model
# Train a logistic regression model
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = trainData, family = binomial)
model
#comment on Coefficients Interpretation:
#Intercept (16.414735): This is the baseline log-odds of survival when all predictors are zero. Since most predictors are categorical, the interpretation is not straightforward.
#Pclass2 (-0.915465) and Pclass3 (-2.334965): These coefficients are negative, indicating that passengers in 2nd and 3rd class had lower odds of survival compared to those in 1st class. The negative values show that being in a lower class decreased the likelihood of survival.
#Sexmale (-2.924401): This negative coefficient suggests that males had lower odds of survival compared to females.
#Age (-0.037960): The negative coefficient indicates that older passengers had slightly lower odds of survival.
#SibSp (-0.290792) and Parch (-0.160382): The negative coefficients suggest that having more siblings/spouses or parents/children aboard slightly decreased the odds of survival.
#Fare (0.001381): The positive coefficient indicates that paying a higher fare slightly increased the odds of survival, though the effect size is very small.
#EmbarkedC (-11.820194), EmbarkedQ (-12.229806), and EmbarkedS (-12.647528): These coefficients are surprisingly large and negative, suggesting an extremely low likelihood of survival for passengers who embarked from these ports. However, such large coefficients might indicate issues with the model or data, such as sparse data for these categories.

# Summary of the model
summary(model)
#comment: AIC stands for Akaike Information Criterion, which evaluates the model's quality by balancing goodness of fit and complexity (number of predictors). Lower AIC values indicate a better model.
#An AIC of 542.5 means the model is relatively good. Lower AIC is better, as it means the model is both accurate and not overly complex.


#6 Random Forest 
# Load the randomForest library
#install.packages("randomForest")
library(randomForest)
# Train a random forest model
rf_model <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = trainData, importance = TRUE, ntree = 500)
# Summary of the model
print(rf_model)

#7 Evaluate the Model
# Predictions on the test data
predictions <- predict(rf_model, testData)
predictions
# Confusion matrix to evaluate the model
confusionMatrix(predictions, testData$Survived)

#Comment: 
#confusion matrix
#            Reference
#Prediction   0   1
#          0 145  32
#          1  19  70
#1 True Positives (TP): 145 - Correctly predicted survivors who actually survived.
#2 True Negatives (TN): 70 - Correctly predicted non-survivors who actually did not survive.
#3 False Positives (FP): 19 - Incorrectly predicted survivors who did not survive.
#4 False Negatives (FN): 32 - Incorrectly predicted non-survivors who actually survived.          
          
#Accuracy: 0.8083 - The overall proportion of correct predictions (both survivors and non-survivors). This means the model correctly predicted survival status 80.83% of the time.          

#Sensitivity and Specificity:
#Sensitivity (Recall): 0.8841 - The proportion of actual survivors correctly identified by the model (true positive rate). This indicates the model is good at identifying survivors.
#Specificity: 0.6863 - The proportion of actual non-survivors correctly identified (true negative rate). This is lower than sensitivity, indicating the model is less effective at identifying non-survivors.
          
#Predictive Values:
#Positive Predictive Value (Precision): 0.8192 - The proportion of predicted survivors that were actually survivors. This indicates that when the model predicts survival, it is correct 81.92% of the time.
#Negative Predictive Value: 0.7865 - The proportion of predicted non-survivors that were actually non-survivors. This indicates that when the model predicts non-survival, it is correct 78.65% of the time.
          
#Other Metrics:
#Prevalence: 0.6165 - The proportion of the dataset that are actual survivors.
#Detection Rate: 0.5451 - The proportion of actual survivors correctly detected by the model.
#Detection Prevalence: 0.6654 - The proportion of instances where the model predicted survival.
#Balanced Accuracy: 0.7852 - The average of sensitivity and specificity, providing a balanced measure of the model’s performance across both classes.
          
          