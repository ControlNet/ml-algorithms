source("model.R")
library(dplyr)
library(Matrix)

Perceptron <- setRefClass("Perceptron", contains = "Classifier",
                          fields = list(w = "Matrix", b = "numeric", history = "data.frame"), methods = list(
    fit = function(x, y, epochs, batch_size = 1, validation_data = NULL, learning_rate_decay = 1,
                   learning_rate = 0.01, shuffle = TRUE, history_per_step = FALSE) {
      # divide the data as batches
      batches <- .self$preprocess(x, y, batch_size, shuffle)

      fit_each_epoch <- function(epoch) {
        fit_each_batch <- function(batch) {
          # reformat the x y dataframe as matrix for computation
          if (batch_size == 1) {
            x_batch <- t(Matrix(batch$x))
            y_batch <- t(Matrix(batch$y))
          } else {
            x_batch <- Matrix(batch$x)
            y_batch <- Matrix(batch$y)
          }
          # predict the data is belonged to which class
          y_hat <- .self$predict(x_batch, .self$max_activation)
          # inverse one hot to find real labels
          y_target <- .self$max_activation(y_batch)
          # locate the misclassified index
          misclassified_index <- (y_hat != y_target) %>% as.vector
          # training, if all the prediction match, skip.
          if (sum(misclassified_index) != 0) {
            # vectorized calculation of the gradient by grad(w) = xy, and grad(b) = y
            grad_w <- (-1 * t(x_batch[misclassified_index, , drop = FALSE]) %*%
              y_batch[misclassified_index, , drop = FALSE]) / sum(misclassified_index)
            grad_b <- -1 * colMeans(y_batch[misclassified_index, , drop = FALSE] %>% Matrix)

            # apply the gradient descent
            .self$w <- .self$w - learning_rate * grad_w
            .self$b <- .self$b - learning_rate * grad_b
          }

          # generate history data for weights and error of given test data
          if (is.null(validation_data)) c(.self$b, as.vector(.self$w))
          else {
            x_test <- validation_data[[1]]
            y_test <- validation_data[[2]]
            test_error <- .self$loss(x_test, y_test)
            # stack the b and w, as the weights history for output
            weights <- rbind(.self$b, .self$w)
            # flatten the matrix as a vector and concat the test_error for output
            c(as.vector(weights), test_error)
          }
        }
        # if the learning_rate_decay is set,
        # here it can apply learning rate decay.
        if (learning_rate_decay != 1) {
          learning_rate <<- learning_rate * learning_rate_decay
        }
        # apply the function above and return the history of this epoch
        batch_history <- lapply(batches, fit_each_batch)
        # output the history by each epoch or by each step(mini-batch)
        if (history_per_step) batch_history
        else batch_history[[length(batch_history)]]
      }
      # reformat the history data frame
      if (history_per_step) {
        # get history from training on each epoch, and combine as a data frame
        weight_history <- 1:epochs %>% lapply(fit_each_epoch) %>%
          unlist(recursive = FALSE) %>% Reduce(f = rbind) %>% as.data.frame
        # generate column names and set
        col_names <- c("step", .self$get_history_names(x, y, validation_data))
        # add step/epoch and colnames for the history data
        .self$history <- cbind(1:nrow(weight_history), weight_history) %>% `colnames<-`(col_names)
        .self
      } else {
        # reformat the history data if only logged on epoch
        weight_history <- as.data.frame(t(sapply(1:epochs, fit_each_epoch)))
        col_names <- c("epoch", .self$get_history_names(x, y, validation_data))
        .self$history <- cbind(1:epochs, weight_history) %>% `colnames<-`(col_names)
        .self
      }
    },

    preprocess = function(x, y, batch_size, shuffle = TRUE) {
      # this function is for preprocessing the data frame input
      x_len <- nrow(x)
      if (!class(y) %in% c("data.frame", "matrix", "Matrix")) y <- Matrix(y)
      n_classes <- ncol(y)
      # if fit in the first time, initialize the weights
      if (.self$w %>% length == 0) {
        .self$w <- Matrix(runif(ncol(x) * n_classes), ncol = n_classes)
        .self$b <- runif(n_classes)
      }
      # if shuffle applied, shuffle the data
      if (shuffle) {
        index <- sample(1:x_len, x_len, replace = FALSE)
        x <- x[index,]
        y <- y[index,]
      }
      # divide the data
      batches <- .self$to_batches(x, y, batch_size)
      batches
    },

    predict = function(x, activation) {
      if (class(x) == "data.frame") x <- x %>% data.matrix %>% Matrix
      else if (class(x) == "numeric") x <- x %>% Matrix %>% t
      # broadcast b
      b <- .self$b %>%
        rep(times = nrow(x)) %>%
        Matrix(nrow = nrow(x), byrow = TRUE)
      # predict the y_hat as f(wx + b) by vectorization computation
      z <- x %*% .self$w + b
      activation(z)
    },

    step_activation = function(z) {
      # this activation function input float(double) numbers, and return -1 or 1
      activation_f <- function(each_z) {
        if (each_z >= 0) 1
        else -1 }
      apply(z, 2, function(row) sapply(row, activation_f))
    },

    max_activation = function(z) {
      # this activation function return the prediction of class with maximum value
      if (ncol(Matrix(z)) != 1) apply(z, 1, which.max)
      # if the model is for binary classification, use the step_activation for the output
      else .self$step_activation(z)
    },

    loss = function(x, y) {
      # this function is for computing the loss with the x y data given
      y_hat <- .self$predict(x, activation = .self$max_activation)
      y_true <- {
        if (ncol(Matrix(y)) != 1) .self$max_activation(y)
        else y
      }
      misclassified_index <- y_true != y_hat
      sum(misclassified_index) / nrow(x)
    },

    get_history_names = function(x, y, validation_data = NULL) {
      # this function for generate the column names of history data
      classes_n <- ncol(Matrix(y))
      # generate weight column names
      col_names <- {
        if (classes_n == 1) {
          w_names <- sapply(1:ncol(x), function(i) paste0("w", i))
          c("b", w_names)
        } else {
          comb_names <- NULL
          for (class_n in 1:classes_n) {
            w_names <- sapply(1:ncol(x), function(i) paste0("c", class_n, "w", i))
            comb_names <- c(comb_names, paste0("c", class_n, "b"), w_names)
          }
          comb_names
        }
      }
      # generate test_error column names if needed
      if (is.null(validation_data)) col_names
      else c(col_names, "test_error")
    }
  )
)
# from `A1_Q5 codebase.R`
read_data <- function(fname, sc) {
  data <- read.csv(file = fname, head = TRUE, sep = ",")
  nr = dim(data)[1]
  nc = dim(data)[2]
  x = data[1:nr, 1:(nc - 1)]
  y = data[1:nr, nc]
  if (isTRUE(sc)) {
    x = scale(x)
    y = scale(y)
  }
  return(list("x" = x, "y" = y))
}
train_data <- read_data("ass1/Task1D_train.csv", FALSE)
x_train <- train_data$x
y_train <- train_data$y

test_data <- read_data("ass1/Task1D_test.csv", FALSE)
x_test <- test_data$x
y_test <- test_data$y

perceptron_one_hot <- function(y, classes = unique(y)) {
  1:length(classes) %>%
    lapply(FUN = function(i) ifelse(y == classes[i], 1, -1) %>% matrix) %>%
    Reduce(f= cbind)
}

y_test_onehot <- perceptron_one_hot(y_test)
y_train_onehot <- perceptron_one_hot(y_train)

set.seed(42)

perceptron1 <- Perceptron()$fit(x_train, y_train_onehot, epoch = 50, batch_size = 5, learning_rate_decay = 0.95,
                               validation_data = list(x_train, y_train_onehot),
                               learning_rate = 0.01, shuffle = TRUE, history_per_step = TRUE)

perceptron2 <- Perceptron()$fit(x_train, y_train_onehot, epoch = 50, batch_size = 5, learning_rate_decay = 0.95,
                               validation_data = list(x_train, y_train_onehot),
                               learning_rate = 0.09, shuffle = TRUE, history_per_step = TRUE)

error_df <- cbind(perceptron1$history$step, perceptron1$history$test_error, perceptron2$history$test_error) %>%
  as.data.frame %>% `colnames<-`(c("step", "model1", "model2"))

ggplot(error_df) +
    geom_line(mapping = aes(x = step, y = model1, color = "learning rate = 0.01"), alpha = 0.8) +
    geom_line(mapping = aes(x = step, y = model2, color = "learning rate = 0.09"), alpha = 0.8) +
    ggtitle("The test error against the mini-batch for 2 learning rates") +
    xlab("mini-batch") + ylab("test error")