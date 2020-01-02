% this is a standard ABC-MCMC (with generic numsimABC) without resampling
rng(500)

nobs = 1000;
sigma = 1;
data = randn(nobs,sigma); % data are iid Gaussian with mean 0 and variance 1
% asume we wish to make inference for the mean of the observations. Assume
% a conjugate (Gaussian) prior.
hyper_mu = 0.1;     % the prior mean 
hyper_sigma = 0.2;  % the prior standard deviation

sobs = mean(data)
nsummary = length(sobs);
collectedsummaries = [];
acceptedsummaries = [];

numsimABC = 2;
mu_start = 0;
R_mcmc = 10000;
burnin = 1000;

mu_old = mu_start;
MCMC = zeros(R_mcmc,1);
MCMC(1) = mu_old;

simdata = mu_old + randn(nobs,numsimABC);
simsumm = mean(simdata,1);
distance = sqrt((simsumm-sobs).^2);
delta = 3e-5;  % was 1e-4
collectedsummaries = [collectedsummaries;simsumm'] ;
% delta = prctile(distance,quantile);
%loglike_old =  log(sum(distance < delta));
loglike_old = -nsummary*log(delta) +logsumexp(-distance.^2/(2*delta^2));
logprior_old = -log(hyper_sigma) - 0.5*(mu_old-hyper_mu)^2 / hyper_sigma^2;

accept=0;
propose=0;
for mcmc_iter = 2:R_mcmc
  propose=propose+1;
  mu = mu_old + 0.1*randn;
  simdata = mu + randn(nobs,numsimABC);  % simulated data
  simsumm = mean(simdata,1);             % summary of simulated data
  if mcmc_iter < burnin
     collectedsummaries = [collectedsummaries;simsumm'] ;
  end
  distance = sqrt((simsumm-sobs).^2);
%  index_inclusion1 = distance < delta;
%  n1 = sum(index_inclusion1);
  %loglike =  log(sum(distance < delta));
  loglike = -nsummary*log(delta) +logsumexp(-distance.^2/(2*delta^2));
  logprior = -log(hyper_sigma) - 0.5*(mu-hyper_mu)^2 / hyper_sigma^2;
  if log(rand) < loglike-loglike_old + logprior - logprior_old
      accept=accept+1;
      MCMC(mcmc_iter) = mu;
      mu_old = mu;
      loglike_old = loglike;
      logprior_old = logprior;
    %  n1
      if mcmc_iter > burnin
         acceptedsummaries = [acceptedsummaries;simsumm'] ;
      end
     % delta = min(delta,prctile(distance,quantile));
  else
      MCMC(mcmc_iter) = mu_old;
  end
end
acceptrate = accept/propose
save('pmABC-MCMC.txt','MCMC','-ascii')

figure
plot(MCMC);
titlestring = sprintf('numsimABC = %d, delta = %d',numsimABC,delta);
title(titlestring)
%legend('ABC')
figure
ksdensity(MCMC(burnin:end));
hold on
% plot EXACT POSTERIOR using a conjugate Gaussian prior for mu
posterior_mean = 1/(1/hyper_sigma^2 + nobs/sigma^2) * (hyper_mu/hyper_sigma^2 + sum(data)/sigma^2);
posterior_std = (1/hyper_sigma^2 + nobs/sigma^2)^(-0.5);
%posterior_draws = posterior_mean + posterior_std * randn(10000,1);
x = [-0.15:.0001:0.15];
y = normpdf(x,posterior_mean,posterior_std);
plot(x,y)
%ksdensity(posterior_draws)
legend('ABC','Exact Bayes')

fprintf('\nABC inference for MU')
[mean(MCMC(burnin:end)), prctile(MCMC(burnin:end),[2.5 97.5])]
fprintf('\nExact inference for MU')
[posterior_mean,posterior_mean-1.96*posterior_std,posterior_mean+1.96*posterior_std]
%prctile(posterior_draws,[2.5 50 97.5])


