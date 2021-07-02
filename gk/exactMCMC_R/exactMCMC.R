library(gk)

data <- read.table("data_2000obs")
log_prior = function(theta){
  A = theta[1]
  B = theta[2]
  g = theta[3]
  k = theta[4]
  
  logA_prior      = dunif(A, -30,30,log=TRUE)
  logB_prior      = dunif(B, 0,30,log=TRUE)
  logg_prior      = dunif(g, 0,30,log=TRUE)
  logk_prior      = dunif(k, 0,30,log=TRUE) 
  
  logprior = logA_prior+logB_prior+logg_prior+logk_prior
}
out = mcmc(x=t(data),N=20000,model=c("gk"),logB=FALSE,get_log_prior = log_prior,theta0=c(3,1,2,0.5), Sigma0=0.1*diag(4), epsilon=1e-06,silent = TRUE)
write(t(out),file="chains",ncolumns=4)
