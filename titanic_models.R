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

test_clean <- test %>%
  group_by(Pclass,
           Sex,
           Embarked) %>%
  mutate(age_missing = is.na(Age),
         Age = coalesce(Age,mean(Age, na.rm=TRUE)),
         Fare = coalesce(mean(Fare, na.rm=TRUE))) %>%
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
test_sparse <- sparse.model.matrix( ~ .-1, data = test_clean)

dtrain <- xgb.DMatrix(data = train_sparse, label = train_clean$Survived) 
dtest <- xgb.DMatrix(data = test_sparse)

# cv for hyperparams

# define the hyperparams
eta <- data.frame(eta = c(0.01))
gamma <- data.frame(gamma = c(0.3))
max_depth <- data.frame(max_depth = c(11))
min_child_weight <- data.frame(min_child_weight = c(4))
subsample <- data.frame(subsample = c(0.68))
colsample_bytree <- data.frame(colsample_bytree = c(0.8))
alpha <- data.frame(alpha = c(0.01))

# build the hyperparams dataset
params <- Reduce(function(x,y) full_join(x, y %>% mutate(foo = 1)),
                 x = list(gamma,max_depth,min_child_weight,subsample,colsample_bytree,alpha),
                 init = eta %>% mutate(foo = 1)) %>%
  select(-foo)

# the hyperparameter search function
xgb_cv <- function(data,params){
  
  set.seed(27)
  
  cv_output <- function(data,params) {
    
   cv_results <- xgb.cv(data = data,
           nrounds = 1000, 
           nthread = 2, 
           nfold = 5, 
           metrics = list("auc"),
           max_depth = params$max_depth, 
           eta = params$eta,
           gamma = params$gamma,
           subsample = params$subsample,
           colsample_bytree = params$colsample_bytree,
           alpha = params$alpha,
           objective = "binary:logistic",
           early_stopping_rounds = 50,
           verbose = FALSE,)
   
   cv_results$evaluation_log[cv_results$best_iteration,] %>%
     mutate(best_iteration = cv_results$best_iteration)
   
  }
  
  params %>%
    rowwise() %>%
    do(data.frame(.,cv_output(data,.)) )
    
}

cv <- xgb_cv(dtrain,params) %>%
  arrange(desc(test_auc_mean))

cv 

cv$evaluation_log[cv$best_iteration,]

set.seed(27)

xgb_model <- xgb.train(data = dtrain,
          nrounds = 20000,
          eta = 0.0005,
          gamma = params$gamma,
          max_depth = params$max_depth,
          min_child_weight = params$min_child_weight,
          subsample = params$subsample,
          colsample_bytree = params$colsample_bytree,
          alpha = params$alpha,
          early_stopping_rounds = 500,
          print_every_n = 100,
          objective = "binary:logistic",
          metrics = list("auc"),
          eval_metric = "auc",
          watchlist = list(train = dtrain))

xgb_importance <- xgb.importance(colnames(train_sparse),
               model = xgb_model,
#               data = train_sparse,
#               label = train_clean$Survived
               )

xgb.plot.importance(xgb_importance)



xgb_roc <- roc(train_clean$Survived ~ predict(xgb_model,dtrain), 
               plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
               print.auc=TRUE, show.thres=TRUE )

test %>%
  ungroup() %>%
  mutate(Survived = ifelse(predict(xgb_model,dtest) > 0.5,1,0)) %>% 
  select(PassengerId,
         Survived) %>%
  write_csv(path = "~/kaggle/titanic/xgb_pred.csv")
  
