# Data Generation
## Libraries:
library(mvtnorm) # generates multivariate Gaussian sampels and calculate the densities
library(ggplot2)
library(reshape2)
source("logistic_regression.R")
## Initialization
set.seed(123)
N <- 1000
c0 <- '+1'; c1 <- '-1' # class labels
mu0 <- c(4.5, 0.5); p0 <- 0.60
mu1 <- c(1.0, 4.0); p1 <- 1 - p0
sigma <- matrix(c(1, 0, 0, 1), nrow=2, ncol=2, byrow = TRUE) # shared covariance matrix
sigma0 <- sigma;   sigma1 <- sigma
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
train.data <- data[train.index, c('x1', 'x2')]; train.label <- data[train.index, 'label']
test.data <- data[-train.index, c('x1', 'x2')]; test.label <- data[-train.index, 'label']

# Initializations
tau.max <- 1000 # maximum number of iterations
eta <- 0.01 # learning rate
epsilon <- 0.01 # a threshold on the cost (to terminate the process)
tau <- 1 # iteration counter
terminate <- FALSE

## Just a few name/type conversion to make the rest of the code easy to follow
X <- as.matrix(train.data) # rename just for conviniance
T <- ifelse(train.label==c0,0,1) # rename just for conviniance

logistic_regression <- LogisticRegression()$fit(X, T, epochs = 5, batch_size = 5,
                                                validation_data = list(X, T), learning_rate_decay = 0.95,
                                                learning_rate = 0.01, shuffle = TRUE, history_per_step = TRUE)

ggplot(data=logistic_regression$history, aes(x=step, y=log(test_error)), color=black) +
    geom_line() + ggtitle('Log of Cost over time') + theme_minimal()

ggplot(data = logistic_regression$history, aes(x = step)) +
  geom_line(aes(y = w1, color = "w1")) +
  geom_line(aes(y = w2, color = "w2")) +
  geom_line(aes(y = b, color = "b")) +
  theme_minimal()

# visualize:
ggplot(data=train.data,
       aes(x=x1, y=x2, label=ifelse(train.label!=c1, '+', '-'),
           color=factor((logistic_regression$predict(X) == T) %>% as.vector))) +
    geom_text(alpha=0.75) +
    scale_color_discrete(guide = guide_legend(title = 'Prediction'))+
    geom_abline(intercept=-logistic_regression$b/logistic_regression$w[2],
                slope=-logistic_regression$w[1]/logistic_regression$w[2]) +
    ggtitle('Training Dataset and Decision Boundary') +
    theme_minimal()