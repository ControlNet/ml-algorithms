source("model.R")
library(dplyr)

KNNModel <- setRefClass("KNNModel", contains = "Classifier",
                        fields = list(k = "numeric", x_train = "data.frame", y_train = "data.frame"), methods = list(
    initialize = function(k) {
      .self$k <- k
    },

    fit = function(x_train, y_train) {
      .self$x_train <- x_train
      .self$y_train <- y_train
      .self
    },

    predict = function(x_test) {
      # define a inner function `majority`
      majority <- function(x) {
        uniqx <- unique(x)
        uniqx[which.max(tabulate(match(x, uniqx)))]
      }
      # define a inner function for applying each row
      predict_for_row <- function(x_test_row) {
        # calculate the distance for each test data
        distance <- .self$x_train %>%
          apply(1, function(x_train_row) {
            x_train_row["dist"] <- sqrt(sum((x_train_row - x_test_row)^2))
            x_train_row
          }) %>%
          t %>%
          as.data.frame %>%
          .["dist"]
        # find the K nearest neighbours' labels
        nearest_indexes <- order(distance$dist)[1:k]
        train_labels <- .self$y_train[nearest_indexes,]
        # predict the test labels with the majority of nearest neighbours
        y_pred_row <- majority(train_labels)
        y_pred_row
      }

      y_pred <- apply(x_test, 1, predict_for_row)
      y_pred
    }
  )
)

KNNRegression <- setRefClass("KNNRegression",
                             contains = "Regressor",
                             fields = list(k = "numeric",
                                           x_train = "data.frame",
                                           y_train = "data.frame"),
                             methods = list(
                               initialize = function(k) {
                                 .self$k <- k
                               },

                               fit = function(x_train, y_train) {
                                 .self$x_train <- x_train
                                 .self$y_train <- y_train
                                 .self
                               },

                               predict = function(x_test) {
                                 # define a inner function for applying each row
                                 predict_for_row <- function(x_test_row) {
                                   # calculate the distance for each test data
                                   distance <- .self$x_train %>%
                                     apply(1, function(x_train_row) {
                                       # Manhattan distance function
                                       x_train_row["dist"] <- sum(abs(x_train_row - x_test_row))
                                       x_train_row
                                     }) %>%
                                     t %>%
                                     as.data.frame %>%
                                     .["dist"]

                                   # find the K nearest neighbours' labels
                                   nearest_indexes <- order(distance$dist)[1:k]
                                   train_labels <- .self$y_train[nearest_indexes,]
                                   # predict the test labels with the mean of nearest neighbours
                                   y_pred_row <- mean(train_labels)
                                   y_pred_row
                                 }

                                 y_pred <- apply(x_test, 1, predict_for_row)
                                 y_pred
                               }
                             )
)