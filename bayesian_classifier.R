library(dplyr)
library(Matrix)
library(mvtnorm)
source("model.R")

BayesianClassifier <- setRefClass("BayesianClassifier", contains = "Classifier",
                                  fields = c("p0", "p1", "mu0", "mu1", "sigma"), methods = list(
  fit = function(x, y) {
    # calculate p(c) for 2 classes
    .self$p0 <- 1 - sum(y) / length(y)
    .self$p1 <- 1 - .self$p0
    # calculate mu for 2 classes
    .self$mu0 <- colMeans(x[!y,])
    .self$mu1 <- colMeans(x[y,])
    # calculate the shared variance
    sigma0 <- var(x[!y,])
    sigma1 <- var(x[y,])
    .self$sigma <- p0 * sigma0 + p1 * sigma1
    .self
  },

  predict = function(x) {
    posterior0 <- .self$p0 * dmvnorm(x = x, mean = .self$mu0, sigma = .self$sigma)
    posterior1 <- .self$p1 * dmvnorm(x = x, mean = .self$mu1, sigma = .self$sigma)
    ifelse(posterior0 > posterior1, 0, 1)
  },

  loss = function(x, y) {
    y_hat <- .self$predict(x)
    sum(y != y_hat) / length(y)
  }
))