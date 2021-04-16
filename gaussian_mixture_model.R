library(dplyr)
library(mvtnorm)

source("model.R")

GaussianMixtureModel <- setRefClass("GaussianMixtureModel", contains = "Clusterer",
                                    fields = list(k = "numeric", d = "numeric",
                                                  epochs = "numeric", mu = "matrix",
                                                  sigma = "matrix", p = "numeric"), methods = list(
    initialize = function(k) {
      .self$k <- k
    },
    fit = function(x, epochs, mode = "soft") {
      x <- as.matrix(x)
      # X dimensions
      .self$d <- ncol(x)
      # initialize parameters
      .self$mu <- matrix(runif(.self$k * .self$d), nrow = .self$k, ncol = .self$d)
      .self$sigma <- matrix(nrow = .self$k, ncol = .self$d ^ 2)
      for (each_k in 1:.self$k) .self$sigma[each_k,] <- diag(.self$d)
      .self$p <- rep(1 / .self$k, times = .self$k)
      # initialize the posterior
      post <- matrix(numeric(), nrow = nrow(x), ncol = .self$k)
      for (epoch in 1:epochs) {
        mu_old <- .self$mu
        # E step
        for (each_k in 1:.self$k) {
          post[,each_k] <-
            dmvnorm(x %>% as.matrix, .self$mu[each_k,], matrix(.self$sigma[each_k,], ncol = .self$d)) * .self$p[each_k]
        }
        # normalization
        post <- post / rowSums(post)
        if (mode == "hard") {
          max_ind <- post == apply(post, 1, max)
          post[max_ind] <- 1
          post[!max_ind] <- 0
        }

        # M step
        # calculate the p(C_k)
        .self$p <- colSums(post) / nrow(post)
        for (each_k in 1:.self$k) {
          # calculate the mu_k
          .self$mu[each_k,] <- ((post[,each_k] * x) %>% colSums) / (.self$p[each_k] * nrow(post))
          # calculate the sigma_k
          .self$sigma[each_k,] <- (t(X - matrix(.self$mu[each_k,], nrow = nrow(x), ncol=.self$d, byrow = TRUE)) %*%
            (post[,each_k]*(X - matrix(.self$mu[each_k,], nrow = nrow(x), ncol=.self$d, byrow = TRUE)))) /
            (.self$p[each_k] * nrow(post))
        }

        if (all.equal(mu_old, .self$mu) == TRUE) break
      }
      print(epoch)
      post
    }
  ))