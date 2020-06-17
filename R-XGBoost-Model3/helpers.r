library(plyr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(xgboost)
library(GGally)
library(pracma)

##' Relative Mean Absolute Error to be used by XGBoost to estimate out-of-sample error (k-fold CV error)
##' @param preds a vector of predicted values
##' @param dtrain data for training in the specified format for XGBoost
relative_mae <- function(preds, dtrain) {
  ## extract labels
  labels <- getinfo(dtrain, "label")
  ## computes error
  err <- mean(abs(preds/labels - 1))
  return(list(metric = "error", value = err))
}


##' Hyperparameter tuning function
##' Wrapper for XGBoost; trains boosted tree models for a (internally specified) set of hyperparameters and maximum number of iterations 1000
##' Produces a plot per hyperparameter combination to be evaluated later by the user
##' Prints training messages
##' @param df_train_regime training split in data.frame format
##' @param regime_name name of regime being run -- are needed in order to place error plots per iteration in the right folder
hyperpar_tuning_energy_model <- function(df_train_regime, regime_name = ""){
  
  ## hyperparameters for tuning via grid search
  hyperpars <- expand.grid(eta = c(0.01, 0.05, 0.1), 
                           gamma = c(0, 1, 5), 
                           max_depth = c(3, 6, 9), 
                           colsample_bytree = 0.8, 
                           subsample = 0.8)
  
  ## transforms dataframe to XGBoost data format
  dtrain <- xgb.DMatrix(as.matrix(df_train_regime %>% dplyr::select(-y, -flight, -time)), 
                        label = as.vector(df_train_regime$y))
  
  
  ## TODO: at some point, nfold should divide per flight. use "folds" argument. 
  ## initialize empty results object
  df_results <- matrix(NA, nrow = nrow(hyperpars), ncol = 8)
  
  ### for each combination of hyperparameter, run 5-fold CV
  list_results <- list()
  for(idx in 1:nrow(hyperpars)){ 
    cat(paste("\nstarting iteration ", idx, " \n", sep = ""))
    
    params <- list(booster = "gbtree", 
                   nthread = 3,
                   eta = hyperpars$eta[idx], 
                   gamma = hyperpars$gamma[idx], 
                   max_depth = hyperpars$max_depth[idx], 
                   colsample_bytree = hyperpars$colsample_bytree[idx],
                   subsample = hyperpars$subsample[idx],
                   objective = "reg:squarederror")
    
    mod <- xgb.cv(params = params, 
                  data = dtrain, 
                  nrounds = 1000,
                  nfold = 5, 
                  showsd = TRUE,
                  early_stopping_rounds = 50,
                  feval = relative_mae,  ## our error function
                  maximize = FALSE, 
                  #metrics = "rmse",#c("rmse", "logloss"), 
                  verbose = TRUE, 
                  print_every_n = 10)
    
    list_results[[idx]] <- as.matrix(mod$evaluation_log)
  }
  
  ## lowest value of CV error - lower bound
  inf_y <- laply(list_results, function(el){
    el <- as.data.frame(el)
    out <- min(el$test_error_mean)
  })
  
  ## highest value of CV error - upper bound
  sup_y <- laply(list_results, function(el){
    el <- as.data.frame(el)
    out <- max(el$test_error_mean)
  })
  
  ## creates error plot per hyperparameter combination: iteration vs error, train and test
  for(i in 1:nrow(hyperpars)){
    df_plot <- as.data.frame(list_results[[i]])
    filename = paste("plotsModels/cruise/hyperpar", i, ".png", sep = "")
    ggplot(df_plot,aes(x=iter))+
      #  geom_point(aes(y=train_rmse_mean), col="black") + 
      #  geom_point(aes(y=test_rmse_mean), col = "red") + 
      geom_line(aes(y=train_error_mean), col="black") + 
      geom_line(aes(y=test_error_mean), col = "red") + 
      geom_line(aes(y=train_error_mean + train_error_std), col = "black", lty = 2) + 
      geom_line(aes(y=test_error_mean + test_error_std), col = "red", lty = 2) + 
      geom_line(aes(y=train_error_mean - train_error_std), col = "black", lty = 2) + 
      geom_line(aes(y=test_error_mean - test_error_std), col = "red", lty = 2) + 
      theme_minimal(base_size = 20) + 
      ylim(0, 1) + 
      ggtitle(paste("eta=", hyperpars$eta[i], 
                    " gamma=",hyperpars$gamma[i], 
                    " max_depth=",hyperpars$max_depth[i], sep = ""))
    ggsave(filename = filename)
  }
  
  return()
}


##' Trains final model, applies to test data and saves csv of predicted values + error
##' @param df_train_regime training split in data.frame format
##' @param df_test_regime test split in data.frame format
##' @param params optimal parameters selected in hyperparameter tuning
##' @param nrounds nrounds selected in hyperparameter tuning
##' @param regime_name name of regime being run -- are needed in order to place error plots per iteration in the right folder
train_test_energy_model <- function(df_train_regime, df_test_regime, params, nrounds, regime_name = ""){
  
  ## transforms dataframe to XGBoost data format
  dtrain <- xgb.DMatrix(as.matrix(df_train_regime %>% dplyr::select(-y, -flight, -time)), 
                        label = as.vector(df_train_regime$y))
  ## train final model
  mod <- xgb.train(params = params, 
                   nrounds = nrounds,
                   data = dtrain, 
                   nfold = 5, 
                   showsd = TRUE,
                   feval = relative_mae, 
                   objective = "reg:squarederror",
                   colsample_bytree = 0.8,
                   subsample = 0.8,
                   maximize = FALSE, 
                   verbose = TRUE)
  
  ## compute one error per flight
  df_error <- ddply(df_test_regime, .(flight), function(df_test_one){
    ## transforms dataframe to XGBoost data format
    dtest <- xgb.DMatrix(as.matrix(df_test_one %>% dplyr::select(-y, -flight, -time)), 
                         label = as.vector(df_test_one$y))
    ## predict test data
    predictions <- predict(mod, newdata = dtest)
    ## truncate predictins at 0 since they should be positive
    predictions <- ifelse(predictions<=0, 0, predictions)
    ## integrate predictions for this flight, since area = power
    est_area <- trapz(1:length(predictions), predictions)
    ## integrate real values for this flight to compare to predicted area
    real_area <- trapz(1:length(df_test_one$y), df_test_one$y)
    ## relative error for areas
    relative_error_area <- abs(est_area/real_area - 1)
    ## squared error for areas
    sq_error_area <- (real_area - est_area)^2
    return(c(relative_error_area, sq_error_area))
  })
  
  ## data to plot errors per flight
  dplot <- data.frame(flights = df_test$flight)
  ## point-wise predictions (for EACH point, and NOT one error per flight)
  dtest <- xgb.DMatrix(as.matrix(df_test %>% dplyr::select(-y, -flight, -time)), ### USO TIME?
                       label = as.vector(df_test$y))
  predictions <- predict(mod, newdata = dtest)
  dplot$predictions <- ifelse(predictions<=0, 0, predictions)
  dplot$relative_error <- abs(dplot$predictions/as.vector(df_test$y) - 1)
  dplot$sq_error <- (as.vector(df_test$y) - dplot$predictions)^2
  
  if(FALSE){
    ## plot relative error
    ggplot(df_error, aes(y = V1)) + 
      geom_boxplot() + 
      theme_minimal() + 
      ylab("area considering relative error")
    
    ## plot squared error
    ggplot(df_error, aes(y = V2)) + 
      geom_boxplot() + 
      theme_minimal()+ 
      ylab("area considering quadratic error")
    
    ## plot relative error per flight
    ggplot(dplot, aes(y = relative_error, group = flights)) + 
      geom_boxplot() + 
      theme_minimal()
    
    ## plot square error per flight
    ggplot(dplot, aes(y = sq_error, group = flights)) + 
      geom_boxplot() + 
      theme_minimal()
  }
  ## write csv file with plot of predicted values
  write.csv(dplot, file.path(model3_dir, paste("/predicted-values/predictions_", regime_name, ".csv", sep = "")))
}
