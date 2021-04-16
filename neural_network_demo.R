# Data Generation
## Libraries:
library(mvtnorm) # generates multivariate Gaussian sampels and calculate the densities
library(ggplot2)
library(reshape2)

## Initialization
set.seed(12345)
N <- 500
c0 <- '0'; c1 <- '1'; c2 <- '2' # class labels
mu0 <- c(1.0, 4.0); p0 <- 0.30
mu1 <- c(4.5, 0.5); p1 <- 0.50
mu2 <- c(3.0, -3.0); p2 <- 1 - p0 - p1

sigma <- matrix(c(1, 0, 0, 1), nrow = 2, ncol = 2, byrow = TRUE) # shared covariance matrix
sigma0 <- sigma; sigma1 <- sigma; sigma2 <- sigma

data <- data.frame(x1 = double(), x2 = double(), label = double()) # empty data.frame

## Generate class labels
data[1:N, 'label'] <- sample(c(c0, c1, c2), N, replace = TRUE, prob = c(p0, p1, p2))
## calculate the size of each class
N0 <- sum(data[1:N, 'label'] == c0);
N1 <- sum(data[1:N, 'label'] == c1); N2 <- N - N0 - N1

## Sample from the Gaussian distribution accroding to the class labels and statitics.
data[data[1:N, 'label'] == c0, c('x1', 'x2')] <- rmvnorm(n = N0, mu0, sigma0)
data[data[1:N, 'label'] == c1, c('x1', 'x2')] <- rmvnorm(n = N1, mu1, sigma1)
data[data[1:N, 'label'] == c2, c('x1', 'x2')] <- rmvnorm(n = N2, mu2, sigma2)
data[data[1:N, 'label'] == c2, 'label'] <- c0
## Take a look at the data set
ggplot(data = data, aes(x = x1, y = x2, color = label, label = ifelse(label == c0, '0', '1'))) +
  geom_text(size = 5, alpha = 0.5) +
  ggtitle('Data set') +
  theme_minimal()
N <- nrow(data)
train.len <- round(N / 2)
train.index <- sample(N, train.len, replace = FALSE)
train.data <- data[train.index, c('x1', 'x2')]
test.data <- data[-train.index, c('x1', 'x2')]
train.label <- data[train.index, 'label']
test.label <- data[-train.index, 'label']
# Some conversions:
## rename just for convenience
N <- train.len

## convert data and labels to matrices
X1 <- unname(data.matrix(train.data))
T1 <- as.numeric(train.label)

X2 <- unname(data.matrix(test.data))
T2 <- as.numeric(test.label)


nn <- NeuralNetwork()$fit(X1, T1, k = 3, 20, 0.0001, validation_data = list(X2, T2), learning_rate = 0.1)
nn$history$epoch <- 1:20
 nn$history %>% melt(id.vars = "epoch") %>% ggplot + geom_line(aes(x = epoch, y = value, color = variable))


