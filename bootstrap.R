Bootstrap <- setRefClass("Bootstrap",
                         fields = list(
                           original_dataset = "data.frame",
                           original_size = "numeric",
                           sample_size = "numeric"),
                         methods = list(
                           initialize = function(original_dataset, sample_size) {
                             .self$original_dataset <- original_dataset
                             # get the original size by getting the row of original dataset
                             .self$original_size <- nrow(original_dataset)
                             .self$sample_size <- sample_size
                           },

                           sample = function(times = 1) {
                             # for each time, generate a bootstrapping indexes and concat by rows
                             indexes <- Reduce(rbind, lapply(1:times, function(t) base::sample(x = .self$original_size,
                                                       size = .self$sample_size, replace = TRUE)))
                             # from indexes get data from original dataset
                             result <- apply(indexes, 1, function(indexes) .self$original_dataset[indexes,])
                             result
                           }
                         )
)