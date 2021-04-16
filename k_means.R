library(dplyr)
library(Matrix)

source("model.R")

KMeans <- setRefClass("KMeans", contains = "Clusterer",
                      fields = list(k = "numeric", d = "numeric",
                                    epochs = "numeric", centroid = "Matrix"),
                      methods = list(
  initialize = function(k) {
    .self$k <- k
  },
  fit = function(x, epochs) {
    x <- Matrix(x)
    # X dimensions
    .self$d <- ncol(x)
    # initialize centroids
    .self$centroid <- Matrix(runif(.self$k * .self$d), nrow = .self$k, ncol = .self$d)
    # initialize the group result
    distance <- Matrix(numeric(), nrow = nrow(x), ncol = .self$k)
    for (epoch in 1:epochs) {
      centroid_old <- .self$centroid
      # E step
      for (each_k in 1:.self$k) {
        distance[,each_k] <- apply(x, 1, function(row) {
          (row - .self$centroid[each_k,]) ^ 2 %>% sum
        })
      }
      cluster <- distance %>% apply(1, which.min)

      # M step
      for (each_k in 1:.self$k) {
        .self$centroid[each_k,] <- x[cluster == each_k,] %>% colMeans
      }

      if (all.equal(centroid_old, .self$centroid) == TRUE) break
    }
    cluster
  }
))