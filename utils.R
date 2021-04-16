library(caTools)

train_test_split <- function(data, label_loc, train_ratio = 0.7) {
  sample <- sample.split(data, SplitRatio = train_ratio)
  train <- subset(data, sample == TRUE)
  x_train <- train[-label_loc]
  y_train <- train[label_loc]
  test <- subset(data, sample == FALSE)
  x_test <- test[-label_loc]
  y_test <- test[label_loc]
  list(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, train=train, test=test)
}

## The following structure helps us to have functions with multiple outputs
### credit: https://stat.ethz.ch/pipermail/r-help/2004-June/053343.html
list <- structure(NA,class="result")
"[<-.result" <- function(x,...,value) {
   args <- as.list(match.call())
   args <- args[-c(1:2,length(args))]
   length(value) <- length(args)
   for(i in seq(along=args)) {
     a <- args[[i]]
     if(!missing(a)) eval.parent(substitute(a <- v,list(a=a,v=value[[i]])))
   }
   x
}
