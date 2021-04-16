# Data Generation
## Libraries:
library(mvtnorm) # generates multivariate Gaussian sampels and calculate the densities
library(ggplot2)
library(reshape2)
source("bayesian_classifier.R")

## Initialization
N <- 1000
c0 <- '+1'; c1 <- '-1' # class labels
mu0 <- c(4.5, 0.5); p0 <- 0.60
mu1 <- c(1.0, 4.0); p1 <- 1 - p0
sigma <- matrix(c(1, 0, 0, 1), nrow=2, ncol=2, byrow = TRUE) # shared covariance matrix
sigma0 <- sigma;   sigma1 <- sigma
### an examle of nonshared covariance matrices
#sigma0 <- matrix(c(0.2, 0.2, 0.2, 0.2), nrow=2, ncol=2, byrow = TRUE);   sigma1 <- matrix(c(1, 0, 0, 1), nrow=2, ncol=2, byrow = TRUE)

data <- data.frame(x1=double(), x2=double(), label=factor(levels = c(c0,c1))) # empty data.frame

## Generate class labels
data[1:N,'label'] <- sample(c(c0,c1), N, replace = TRUE, prob = c(p0, p1))

## calculate the size of each class
N0 <- sum(data[1:N,'label']==c0); N1 <- N - N0

## Sample from the Gaussian distribution accroding to the class labels and statitics.
data[data[1:N,'label']==c0, c('x1', 'x2')] <- rmvnorm(n = N0, mu0, sigma0)
data[data[1:N,'label']==c1, c('x1', 'x2')] <- rmvnorm(n = N1, mu1, sigma1)

## Split data to train and test datasets
train.len <- round(N/2)
train.index <- sample(1:N, train.len, replace = FALSE)
train.data <- data[train.index, c('x1', 'x2')]
test.data <- data[-train.index, c('x1', 'x2')]
train.label <- data[train.index, 'label']
test.label <- data[-train.index, 'label']


baysian_classifier <- BayesianClassifier()$fit(train.data, train.label == "-1")
baysian_classifier$loss(train.data, train.label == "-1")
baysian_classifier$loss(test.data, test.label == "-1")
