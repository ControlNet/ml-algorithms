cross_validation <- function(x, y, num_fold, pred_func, error_func) {
  # test data length, which is calculated from num_fold
  test_length <- floor(nrow(x) / num_fold)
  # A function for split the x, y dataset to `num_fold` parts
  divide_data <- function(i) {
    index <- i:(i + test_length - 1)
    list(x=x[index,], y=y[index])
  }
  # I is the indexes of the begining of data blocks
  I <- 1 + (0:(num_fold-1)) * test_length
  # apply the function to get divided data blocks
  data_blocks <- lapply(I, divide_data)

  # the inner function for calculate error for each iteration
  calulate_error <- function(i) {
    # assign the components to variables
    x_test <- data_blocks[[i]]$x
    y_test <- data_blocks[[i]]$y
    train_blocks <- data_blocks[-i]
    x_train_blocks <- lapply(train_blocks, function(each) each$x)
    y_train_blocks <- lapply(train_blocks, function(each) each$y)
    # merge the train data
    x_train <- Reduce(rbind, x_train_blocks)
    y_train <- data.frame(unlist(y_train_blocks))
    # predict the test data to get predicted labels
    y_pred <- pred_func(x_train, y_train, x_test)
    # calculate the error
    error_func(y_pred, y_test)
  }
  mean(sapply(1:num_fold, calulate_error))
}
