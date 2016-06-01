# Dynamic Factor Copula
In order to compile the cpp file and link to R, you need install Rcpp, RcppArmadillo, RcppGSL, openmp as well as GSL library.
A sample of R code could be as the following

```R
load("datagen.RData")

Sys.setenv("PKG_CXXFLAGS"="-fopenmp")
Sys.setenv("PKG_LIBS"="-fopenmp")

library("inline")
library("Rcpp")
library(xtable)

# Compile Cpp file
sourceCpp("parallel_block_modelselection.cpp")

# data list as input: 
#     u_mat is copula data matrix (eg. 1000 * 100)
#     gid is the group id of each series 
#     n_max is the number of columns (eg. 100)
#     t_max is the number of rows (eg. 1000)
#     n_group is the number of group (eg. 10)

data <- list(u=u_mat,gid=gid, n_max=n_max, t_max = t_max ) 
n_group <- max(gid)

# inits as the starting point for MCMC
#     a is a vector size n_group
#     b is a vector size n_group
#     nu is a vector size n_group
#     gamma is a vector size n_group
#     z is a vector of state size t_max
#     zeta is a matrix size t_max * n_group 

zeta_new <- matrix(10, nrow = t_max, ncol = n_group)
nu_new <- sample(15:20,n_group,replace = T)
for (i in 1:n_group){
  zeta_new[,i] <- rinvgamma(n = t_max,shape = nu_true[i]/2, scale = nu_true[i]/2)
}
gamma_new = rep(-0.2,n_group)
inits <- list(a = rep(0.05,n_group), b= rep(0.98,n_group), f0 = rep(1.5,n_group), z = rnorm(t_max), nu = nu_new, zeta = zeta_new, gamma = gamma_new)

# We have 10000 iterations into 1000 batches
iter_num <- 10000
numbatches=1000; batchlength=10;

# To run MCMC 
#     core is the number of CPU
#     disttype is the type of copula (1. Gaussian, 2. Student, 3. Hyperbolic Skew Student)
#     modelselect = F to save time of estimation.

save = MCMCstep(data, inits, iter = iter_num, numbatches=numbatches, 
                batchlength=batchlength, other = list(core=8,disttype = 1,modelselect = F))


```

