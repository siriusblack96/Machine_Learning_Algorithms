################################################ KNN #####################################################
# Load data
load("~/Downloads/class_data.RData")
df <- data.frame(x=x, y=y)

library(class)
set.seed(35)

# Split the data into training and testing sets (70% for training, 30% for testing)
#train_index <- createDataPartition(df$y, p = 0.7, list = FALSE)
train <- df[,]
test <- data.frame(x=xnew)

# Define hyperparameter grid
hyper_grid <- expand.grid(k = 5:15)

# Define train control for cross-validation
folds <- seq(5,30, by = 1)
train_control_list <- list()
for (i in 1:length(folds)) {
  train_control_list[[i]] <- trainControl(method = "cv", number = folds[i], allowParallel = FALSE)
}

# Train the KNN model using cross-validation and hyperparameter tuning for each value of folds
knn.pred_list <- list()
best_k <- 0
best_acc <- 0
best_fold <- 0
for (i in 1:length(train_control_list)) {
  knn.pred_list[[i]] <- train(train[,-ncol(train)], as.factor(train$y), method = "knn", trControl = train_control_list[[i]], tuneGrid = hyper_grid)
  if (max(knn.pred_list[[i]]$results$Accuracy) > best_acc) {
    best_acc <- max(knn.pred_list[[i]]$results$Accuracy)
    best_k <- knn.pred_list[[i]]$bestTune$k
    best_fold <- i+4
  }
}

# Print the best fold and k combination based on the lowest misclassification error rate
cat("\nBest Fold-K Combination (kNN):", best_fold, "folds and k =", best_k, "\n")
print(knn.pred_list[[best_fold-4]])


ynew <- predict(knn.pred_list[[best_fold-4]], newdata = test[,],k=best_k)

test_error <- (1 - best_acc)*100
save(ynew,test_error,file="19.RData")