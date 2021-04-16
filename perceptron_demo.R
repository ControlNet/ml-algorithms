library(MASS) # generates multivariate Gaussian sampels
library(ggplot2)
library(reshape2)
source("perceptron.R")

## Generative parameters
c0 <- '+1'; c1 <- '-1' # class labels
mu0 <- c(4.5, 0.5); p0 <- 0.60
mu1 <- c(1.0, 4.0); p1 <- 1 - p0
sigma <- matrix(c(1, 0, 0, 1), nrow=2, ncol=2, byrow = TRUE) # shared covariance matrix
sigma0 <- sigma;   sigma1 <- sigma
### an examle of nonshared covariance matrices
#sigma0 <- matrix(c(0.2, 0.2, 0.2, 0.2), nrow=2, ncol=2, byrow = TRUE);   sigma1 <- matrix(c(1, 0, 0, 1), nrow=2, ncol=2, byrow = TRUE)

## Initialization
set.seed(123)
N <- 1000
data <- data.frame(x1=double(), x2=double(), label=factor(levels = c(c0,c1))) # empty data.frame

## Generate class labels (Step 1)
data[1:N,'label'] <- sample(c(c0,c1), N, replace = TRUE, prob = c(p0, p1))

## calculate the size of each class
N0 <- sum(data[1:N,'label']==c0); N1 <- N - N0

## Sample from the Gaussian distribution accroding to the class labels and statitics. (Steps 2 & 3)
data[data[1:N,'label']==c0, c('x1', 'x2')] <- mvrnorm(n = N0, mu0, sigma0)
data[data[1:N,'label']==c1, c('x1', 'x2')] <- mvrnorm(n = N1, mu1, sigma1)

## Split data to train and test datasets
train.len <- round(N/2)
train.index <- sample(1:N, train.len, replace = FALSE)
train.data <- data[train.index, c('x1', 'x2')]
test.data <- data[-train.index, c('x1', 'x2')]
train.label <- data[train.index, 'label']
test.label <- data[-train.index, 'label']

# Initialization
eta <- 0.01 # Learning rate
epsilon <- 0.001 # Stoping criterion
tau.max <- 100 # Maximum number of iterations
Phi <- as.matrix(cbind(1, train.data))
T <- ifelse(train.label == c0, eval(parse(text=c0)),eval(parse(text=c1))) # Convention for class labels

perceptron <- Perceptron()$fit(train.data, T, 1, batch_size = 1, validation_data = list(
  x = test.data, y = ifelse(test.label == c0, eval(parse(text=c0)),eval(parse(text=c1)))), learning_rate = 0.01,
  history_per_step = TRUE, shuffle = FALSE)

# perceptron$loss(test.data, ifelse(test.label == c0, eval(parse(text=c0)),eval(parse(text=c1))))
# perceptron$history
ggplot(perceptron$history, aes(x= step, y=test_error)) + geom_line()

ggplot(data = perceptron$history, aes(x = step)) +
  geom_line(aes(y = w1, color = "w1")) +
  geom_line(aes(y = w2, color = "w2")) +
  geom_line(aes(y = b, color = "b")) +
  theme_minimal()

ggplot(data=as.data.frame(Phi), aes(x=x1, y=x2, label=ifelse(T!=c1, '+', '-'),
                                    color = factor(perceptron$predict(train.data, perceptron$step_activation) == T))) +
    geom_text(alpha=0.75) +
    scale_color_discrete(guide = guide_legend(title = 'Prediction'))+
    geom_abline(intercept=perceptron$b[1], slope=-perceptron$w[1]/perceptron$w[2]) +
    ggtitle('Training Dataset and Decision Boundary') +
    theme_minimal()