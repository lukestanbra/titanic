# libraries
library(tidyverse)
library(pROC)
library(caret)
library(xgboost)
library(Matrix)

# load the data
train <- read.csv("~/kaggle/titanic/train.csv")
test <- read.csv("~/kaggle/titanic/test.csv")

summary(train)

train_clean <- train %>%
  group_by(Pclass,
           Sex,
           Embarked) %>%
  mutate(age_missing = is.na(Age),
         Age = coalesce(Age,mean(Age, na.rm=TRUE))) %>%
  select(-PassengerId,
         -Name,
         -Cabin,
         -Ticket)

summary(train_clean)

# logistic regression
logit_model <- glm(data = train_clean, 
      formula = Survived ~ .,
      family = binomial(link = 'logit'))

logit_model_with_interactions <- step(logit_model, ~.^2)

summary(logit_model_with_interactions)

rocobj <- roc(logit_model_with_interactions$data$Survived ~ logit_model_with_interactions$fitted.values ,
    plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
    print.auc=TRUE, show.thres=TRUE)

coords(rocobj, "b", ret="t")

#xgboost
train_sparse <- sparse.model.matrix(Survived ~ .-1, data = train_clean)

dtrain <- xgb.DMatrix(data = train_sparse, label = train_clean$Survived) 

xgb_model <- xgb.train(data = dtrain,
          nrounds = 100)

xgb_importance <- xgb.importance(colnames(train_sparse),
               model = xgb_model,
#               data = train_sparse,
#               label = train_clean$Survived
               )

xgb.plot.importance(xgb_importance)
