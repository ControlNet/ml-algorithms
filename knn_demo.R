library(datasets)
source("utils.R")
source("knn.R")
source("bootstrap.R")
data(iris)

data <- train_test_split(iris, label_loc = 5, train_ratio = 0.7)
train <- data$train
x_train <- data$x_train
y_train <- data$y_train
x_test <- data$x_test
y_test <- data$y_test

train_boot <- Bootstrap(original_dataset = cbind(x_train, y_train), sample_size = 2000)$sample_once()
x_train <- train_boot[-5]
y_train <- train_boot[5]
knn <- KNNModel(k = 19)$fit(x_train = x_train, y_train = y_train)
y_pred <- knn$predict(x_test)

knn$evaluate_accuracy(y_pred, y_test$Species)
table(y_pred, y_test[["Species"]])

