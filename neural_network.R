library(dplyr)
library(Matrix)
source("model.R")
source("utils.R")

# 3-layer neural network
NeuralNetwork <- setRefClass("NeuralNetwork", contains = "Classifier",
                             fields = list(w1 = "Matrix", b1 = "numeric", w2 = "Matrix",
                                           b2 = "numeric", history = "data.frame"), methods = list(
    fit = function(x, y, k, epochs, lambda, validation_data = NULL, learning_rate_decay = 1,
                   learning_rate = 0.01, shuffle = TRUE, history_per_step = FALSE) {
      m <- dim(x)[1] # number of data
      d0 <- dim(x)[2] # input layer dimension
      d1 <- k # hidden layer dimension
      d2 <- 1 # for binary classifier, only 1 output dimension

      # initialize weights
      .self$w1 <- Matrix(0.01 * rnorm(d0 * d1, sd = 0.5), nrow = d0, ncol = d1)
      .self$b1 <- rep(0, times = d1)
      .self$w2 <- Matrix(0.01 * rnorm(d1 * d2, sd = 0.5), nrow = d1, ncol = d2)
      .self$b2 <- rep(0, times = d2)

      # initialize history data frame
      .self$history <- {
        if (is.null(validation_data))
          data.frame(train_acc = numeric())
        else data.frame(train_acc = numeric(), test_acc = numeric())
      }

      if (!is.null(validation_data)) {
        x_test <- validation_data[[1]]
        y_test <- validation_data[[2]]
      }

      fit_each_epoch <- function(epoch) {
        fit_each_data <- function(i) {
          # i-th row
          xi <- x[i, , drop = FALSE]
          yi <- y[i]

          # Feedforward:
          list[a1, a2, z1, z2] <- .self$feedforward(xi)
          # Backpropagation:
          list[de1, de2] <- .self$backpropagation(yi, z1, z2, a2)
          # calculate the derivative
          w1d <- t(xi) %*% de1
          b1d <- de1
          w2d <- t(a1) %*% de2
          b2d <- de2
          # update weights
          .self$w1 <- .self$w1 - learning_rate * (w1d + lambda * .self$w1)
          .self$b1 <- .self$b1 - learning_rate * b1d %>% as.numeric
          .self$w2 <- .self$w2 - learning_rate * (w2d + lambda * .self$w2)
          .self$b2 <- .self$b2 - learning_rate * b2d %>% as.numeric

          # calculate errors, add to history
          if (history_per_step)
            .self$history[nrow(.self$history) + 1,] <- {
              if (is.null(validation_data)) {
                c(train_acc = .self$evaluate(x, y))
              } else {
                c(train_acc = .self$evaluate(x, y), test_acc = .self$evaluate(x_test, y_test))
              }
            }
        }
        1:m %>% sapply(fit_each_data)

        if (!history_per_step) {
          .self$history[nrow(.self$history) + 1,] <- {
            if (is.null(validation_data)) {
              c(train_acc = .self$evaluate_acc(x, y))
            } else {
              c(train_acc = .self$evaluate_acc(x, y), test_acc = .self$evaluate_acc(x_test, y_test))
            }
          }
        }

        if (learning_rate_decay < 1) {
          learning_rate <<- learning_rate * learning_rate_decay
        }
      }
      1:epochs %>% lapply(fit_each_epoch)
      .self
    },

    feedforward = function(x) {
      batch <- dim(x)[1]
      # hidden layer
      z1 <- x %*% .self$w1 + rep(.self$b1, batch) %>% Matrix(byrow = TRUE, nrow = batch)
      a1 <- .self$h(z1)
      # output layer
      z2 <- a1 %*% .self$w2 + rep(.self$b2, batch) %>% Matrix(byrow = TRUE, nrow = batch)
      a2 <- .self$h(z2)
      list(a1, a2, z1, z2)
    },

    backpropagation = function(y, z1, z2, a2) {
      # output layer
      de2 <- -(y - a2) * .self$hd(z2)
      # hidden layer
      de1 <- de2 %*% t(.self$w2) * .self$hd(z1)
      list(de1, de2)
    },

    probability = function(x) .self$feedforward(x)[[2]],
    predict = function(x, threshold = 0.5) ifelse(.self$probability(x) >= threshold, 1, 0),
    evaluate_acc = function(x, y, threshold = 0.5) .self$evaluate(.self$predict(x, threshold), y),

    h = function(z) 1 / (1 + exp(-z)),
    hd = function(z) .self$h(z) * (1 - .self$h(z))
  )
)