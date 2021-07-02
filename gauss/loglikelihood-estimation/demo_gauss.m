% this is a standard ABC-MCMC (with generic numsimABC) without resampling
rng(500)  % set to reproduce the same data

nobs = 1000;
mu_true = 0;
sigma = 1;
data = mu_true + randn(nobs,sigma); % data are iid Gaussian with mean 0 and variance 1
% asume we wish to make inference for the mean of the observations. Assume
% a conjugate (Gaussian) prior.
hyper_mu = 0.1;     % the prior mean 
hyper_sigma = 0.2;  % the prior standard deviation

sobs = mean(data)
nsummary = length(sobs);

numsimABC = 500;
numrepetitions = 1000;
%mu_vec = [-1:.1:1];
mu_vec = linspace(-0.1,0.1,50);   % 50 parameter points to try in the interval [-0.1,0.1]. Notice the interval includes the true value mu=0

loglike_vec_abcmcmc = zeros(numrepetitions,length(mu_vec));

delta = 3e-5;  % was 1e-4;

rng(100)  % set to reproduce the same stream of random numbers when computing loglikelihoods

tic
for mm = 1:length(mu_vec)
    mm
   for rr = 1:numrepetitions

     simdata = mu_vec(mm) + randn(nobs,numsimABC);
     simsumm = mean(simdata,1);
     distance = sqrt((simsumm-sobs).^2);
    % log_constant = -normlike([mu_vec(mm),1],sobs)-(-normlike([mu_vec(mm),1+delta^2],sobs));
    % loglike_vec(rr,mm) = log_constant -nsummary/2*log(2*pi) -nsummary*log(delta) +logsumexp(-distance.^2/(2*delta^2));
    % loglike_vec(rr,mm) = logsumexp(-distance.^2/(2*delta^2));
     loglike_vec_abcmcmc(rr,mm) =  -nsummary*log(delta) -log(numsimABC) + logsumexp(-distance.^2/(2*delta^2));
   end
end
eval=toc
save('loglike_vec_abcmcmc','loglike_vec_abcmcmc')

plot(mu_vec,mean(loglike_vec_abcmcmc,1),'r-',mu_vec,prctile(loglike_vec_abcmcmc,2.5),'r--',mu_vec,prctile(loglike_vec_abcmcmc,97.5),'r--')
hold on

return
% now superimpose the exact loglikelihood for comparison
% but what does it mean "exact likelihood" here?
% of course it makes no sense to compare the ABC likelihood (computed using
% summary statistics) with the exact likelihood based on the full data, as
% these are defined on different spaces. So below we compute the exact
% likelihood of the sample mean of Gaussian observations, which is sobs~N(theta,1/nobs)
loglike_exact = zeros(length(mu_vec),1);
for mm = 1:length(mu_vec) 
  loglike_exact(mm) =log(nobs) -0.5*log(2*pi*nobs) -nobs/2*(sobs-mu_vec(mm))^2;
end
plot(mu_vec,loglike_exact,'g-')

