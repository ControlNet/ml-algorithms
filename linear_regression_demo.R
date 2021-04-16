library(ggplot2)
source("linear_regression.R")

lr_model <- LinearRegression()
set.seed(1234)

# geberate N x D data samples
N <- 200 # number of all data point (test and train)
D <- 4 # number of features/attributes
data <- data.frame(matrix(runif(D * N), nrow = N, ncol = D))

# generate the labels
coeff <- matrix(c(-5, -3, 4, 5, 10), D + 1, 1) # the real coefficient to be estimated
data <- cbind(data, 'Y' = as.matrix(cbind(1, data[, 1:D])) %*% coeff)
# add gaussian noise the labels (just to make it a little more challenging)
data$Y <- data$Y + rnorm(N, mean = 0, sd = 1)
train.len <- N / 2
train.index <- sample(1:N, train.len)
train.data <- data[train.index, 1:D]
train.label <- data[train.index, 'Y']
test.data <- data[-train.index, 1:D]
test.label <- data[-train.index, 'Y']

lr_model <- lr_model$fit(train.data, train.label, epochs = 1000, learning_rate = 0.1, batch_size = 100)
dim(lr_model$history)
lr_model$b
lr_model$w
# l$history
ggplot(data = lr_model$history, aes(x = epoch)) +
  geom_line(aes(y = w1, color = "w1")) +
  geom_line(aes(y = w2, color = "w2")) +
  geom_line(aes(y = w3, color = "w3")) +
  geom_line(aes(y = w4, color = "w4")) +
  geom_line(aes(y = b, color = "b")) +
  theme_minimal()

lr_model$predict(test.data) >= 0