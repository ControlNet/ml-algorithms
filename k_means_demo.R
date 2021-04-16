options(warn=-1)
library(mvtnorm) # generates multivariate Gaussian sampels and calculate the densities
library(ggplot2) # plotting
library(reshape2) # data wrangling!
library(clusterGeneration) # generates the covariance matrices that we need for producing synthetic data.

source("k_means.R")

# Set the parameters:
set.seed(12345) # save the random seed to make the results reproducble
N <- 1000 # number of samples
K <- 3    # number of clusters
D <- 2    # number of dimensions

# Initializations:
Phi <- runif(K); Phi <- Phi/sum(Phi)    # Phi(k) indicates the fraction of samples that are from cluster k
Nk <- matrix(0,nrow = K)    # initiate  the effective number of points assigned to each cluster
Mu <- matrix(runif(K*D, min=-1)*10,nrow = K, ncol = D)    # initiate the centriods (means) of the clusters (randomly chosen)
Sigma <- matrix(0,nrow = K, ncol = D^2)    # initiate the covariance matrix

# Create the covariance matrices:
for (k in 1:K){
    # For each cluster generate one sigma matrix
    Sigma[k,] <- genPositiveDefMat(D)$Sigma[1:D^2]
}

# Generate data:
data <- data.frame(K=integer(), X1=double(), X2=double()) # empty dataset
data[1:N,'K'] <- sample(1:K, N, replace = TRUE, prob = Phi) # geenrate labels (they will not be used in EM, just for validation)
## For each cluster k:
for (k in 1:K){
    ### calculate the effective number of points assigned to it:
    Nk[k] <- sum(data$K==k)
    ### generate the actual points:
    data[data$K==k, 2:3] <- rmvnorm(n = Nk[k], Mu[k,], matrix(Sigma[k,], ncol=D))
}

# Remove the lables! So, our GMM has no clue what are the real labels.
X <- as.matrix(data[,-1])

group <- KMeans(3)$fit(X, 20)

ggplot(X %>% as.data.frame) + geom_point(aes(x=X1, y=X2, color=as.factor(group)))