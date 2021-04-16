Model <- setRefClass("Model", fields = list(), methods = list(
  intialize = function() { },
  fit = function(x_train = NULL, y_train = NULL) { },
  predict = function(x_test) { },
  evaluate = function(y_pred, y_test) { }
))

SupervisedModel <- setRefClass("SupervisedModel", contains = "Model", fields = list(), methods = list(
  to_batches = function(x, y, batch_size) {
    # this function is for divide the data as batches
    # as the batches are not divided equally, so there are complete batches,
    # and a residual batch.
    x <- as.matrix(x)
    y <- as.matrix(y)
    x_nrow <- nrow(x)
    # the number of complete batches
    complete_batch_num <- floor(x_nrow / batch_size)
    # The last row of complete batches
    complete_nrow <- complete_batch_num * batch_size
    # locate the residual part
    if (complete_nrow == x_nrow) {
      x_residual <- NULL
      y_residual <- NULL
      batch_nums <- complete_batch_num
    } else {
      x_residual <- x[(complete_nrow + 1):x_nrow,]
      y_residual <- y[(complete_nrow + 1):x_nrow,]
      batch_nums <- complete_batch_num + 1
    }
    # get residual batch data
    residual_nrow <- x_nrow - complete_nrow
    if (residual_nrow == 1) {
      x_residual <- t(x_residual)
      y_residual <- t(y_residual)
    }
    # locate and get other batches
    lapply(1:batch_nums, function(i) {
      start_index <- (i - 1) * batch_size + 1
      if (start_index + batch_size - 1 <= x_nrow) {
        list(x = x[start_index:(start_index + batch_size - 1),],
             y = y[start_index:(start_index + batch_size - 1),])
      } else {
        list(x = x_residual, y = y_residual)
      }
    })
  }
))

UnsupervisedModel <- setRefClass("UnsupervisedModel", contains = "Model", methods = list(
  to_batches = function(x, batch_size) {
    # this function is for divide the data as batches
    # as the batches are not divided equally, so there are complete batches,
    # and a residual batch.
    x <- as.matrix(x)
    x_nrow <- nrow(x)
    # the number of complete batches
    complete_batch_num <- floor(x_nrow / batch_size)
    # The last row of complete batches
    complete_nrow <- complete_batch_num * batch_size
    # locate the residual part
    if (complete_nrow == x_nrow) {
      x_residual <- NULL
      batch_nums <- complete_batch_num
    } else {
      x_residual <- x[(complete_nrow + 1):x_nrow,]
      batch_nums <- complete_batch_num + 1
    }
    # get residual batch data
    residual_nrow <- x_nrow - complete_nrow
    if (residual_nrow == 1) {
      x_residual <- t(x_residual)
    }
    # locate and get other batches
    lapply(1:batch_nums, function(i) {
      start_index <- (i - 1) * batch_size + 1
      if (start_index + batch_size - 1 <= x_nrow) {
        x[start_index:(start_index + batch_size - 1),]
      } else {
        x_residual
      }
    })
  }
))


Classifier <- setRefClass("Classifier", contains = "SupervisedModel", methods = list(
  evaluate = function(y_pred, y_test) .self$evaluate_accuracy(y_pred, y_test),
  confusionMatrix = function(y_pred, y_test) table(y_pred, y_test),
  evaluate_accuracy = function(y_pred, y_test) sum(y_pred == y_test) / length(y_pred)
))

Regressor <- setRefClass("Regressor", contains = "SupervisedModel", methods = list())

Clusterer <- setRefClass("Clusterer", contains = "UnsupervisedModel", methods = list())