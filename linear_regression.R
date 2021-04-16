source("model.R")

LinearRegression <- setRefClass("LinearRegression", contains = "Regressor",
                                fields = list(w = "matrix", b = "numeric", history = "data.frame"), methods = list(
    fit = function(x, y, epochs, batch_size = nrow(x), learning_rate = 0.01,
                   shuffle = TRUE) {
      x_len <- nrow(x)
      .self$w <- matrix(runif(ncol(x)))
      .self$b <- runif(1)
      if (shuffle) {
        index <- sample(1:x_len, x_len, replace = FALSE)
        x <- x[index,]
        y <- y[index]
      }
      batches <- .self$to_batches(x, y, batch_size)
      fit_each_epoch <- function(epoch) {
        fit_each_batch <- function(batch) {
          x_batch <- {
            if (batch_size == 1) t(matrix(batch$x))
            else batch$x
          }
          y_batch <- matrix(batch$y)
          y_hat <- .self$predict(x_batch)
          error <- y_batch - y_hat
          grad_w <- matrix(-1 * t(error) %*% x_batch) / nrow(x_batch)
          grad_b <- -1 * mean(error)
          .self$w <- .self$w - learning_rate * grad_w
          .self$b <- .self$b - learning_rate * grad_b
          c(.self$b, as.vector(.self$w))
        }
        batch_history <- lapply(batches, fit_each_batch)
        batch_history[[length(batch_history)]]
      }
      weight_history <- as.data.frame(t(sapply(1:epochs, fit_each_epoch)))
      weight_names <- sapply(1:ncol(x), function(i) paste0("w", i))
      colnames(weight_history) <- c("b", weight_names)
      .self$history <- cbind(data.frame("epoch" = 1:epochs), weight_history)
      .self
    },

    predict = function(x) {
      as.matrix(x) %*% .self$w + .self$b
    },

    loss = function(y, y_hat) {
      (y - y_hat)^2 / 2
    },
  )
)
