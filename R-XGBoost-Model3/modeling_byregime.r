library(plyr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(xgboost)
library(GGally)
library(pracma)

root <- rprojroot::has_file(".git/index")
model3_dir <- root$find_file("R-XGBoost-Model3")
source(file.path(model3_dir, 'helpers.R')) # functions for training model

df_all_takeOff <- read.csv(file.path(model3_dir, "../data/data_clean_takeOff.csv"))[,-1]
df_all_cruise <- read.csv(file.path(model3_dir, "../data/data_clean_cruise.csv"))[,-1]
df_all_landing <- read.csv(file.path(model3_dir, "../data/data_clean_landing.csv"))[,-1]

set.seed(1502)
flights <- unique(df_all_takeOff$flight)
flights_train <- read.csv(file.path(model3_dir, "../data/poll.csv"))[,1]

### takeOff
###########
idx_train <- which(df_all_takeOff$flight %in% flights_train)
df_train <- df_all_takeOff[idx_train,]
df_test <- df_all_takeOff[-idx_train,]

hyperpar_tuning_energy_model(df_train, regime_name = "takeOff")

params <- list(nthread = 3,
               eta = 0.05, 
               gamma = 1, 
               max_depth = 3)
nrounds <- 250

train_test_energy_model(df_train, df_test, params, nrounds, regime_name = "takeOff")

###########

### cruise
###########
set.seed(1502)
idx_train <- which(df_all_cruise$flight %in% flights_train)
df_train <- df_all_cruise[idx_train,]
df_test <- df_all_cruise[-idx_train,]

hyperpar_tuning_energy_model(df_train, regime_name = "cruise")

params <- list(nthread = 3,
               eta = 0.05, 
               gamma = 5, 
               max_depth = 6)
nrounds <- 250

train_test_energy_model(df_train, df_test, params, nrounds, regime_name = "cruise")

###########

### landing
###########
idx_train <- which(df_all_landing$flight %in% flights_train)
df_train <- df_all_landing[idx_train,]
df_test <- df_all_landing[-idx_train,]

hyperpar_tuning_energy_model(df_train, regime_name = "landing")

params <- list(nthread = 3,
               eta = 0.01, 
               gamma = 1, 
               max_depth = 6)
nrounds <- 500

train_test_energy_model(df_train, df_test, params, nrounds, regime_name = "landing")

###########