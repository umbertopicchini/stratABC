% this demo impplements ABC with resampling

rng(500)

nobs = 1000;  % MUST BE AN EVEN NUMBER
sigma = 1;
data = randn(nobs,sigma); % data are iid Gaussian with mean 0 and variance 1
% asume we wish to make inference for the mean of the observations. Assume
% a conjugate (Gaussian) prior.
hyper_mu = 0.1;     % the prior mean 
hyper_sigma = 0.2;  % the prior standard deviation

sobs = mean(data)
nsummary = length(sobs);

numsimABC = 1;
numresample1 = 500;

mu_start = 0;
R_mcmc = 10000;
burnin = 1000;

mu_old = mu_start;
MCMC = zeros(R_mcmc,1);
MCMC(1) = mu_old;
delta = 3e-5;

resample_indeces1 = zeros(nobs,numresample1);
for jj=1:numresample1
     indeces = randsample(nobs,nobs,'true');
     resample_indeces1(:,jj) = indeces;
end

simdata1 = mu_old + randn(nobs,numsimABC);
if numsimABC ==1
    % training
    simdata_resampled1 = simdata1(resample_indeces1);  % a nobs x numresample matrix
    simsumm1 = mean(simdata_resampled1,1);  % summary statistics
elseif  numsimABC>1
    error('at the moment NUMSIMABC can only equal 1.')
end
distance = sqrt((simsumm1-sobs).^2);

%like_old = 1/delta^nsummary * sum(exp(-distance.^2/(2*delta^2)));

loglike_old =  -nsummary*log(delta) + logsumexp(-distance.^2/(2*delta^2));
logprior_old = -log(hyper_sigma) - 0.5*(mu_old-hyper_mu)^2 / hyper_sigma^2;

accept=0;
propose=0;
for mcmc_iter = 2:R_mcmc
    propose=propose+1;
    mu = mu_old + 0.1*randn;
  
    simdata1 = mu + randn(nobs,numsimABC);
    if numsimABC ==1
       % training
       simdata_resampled1 = simdata1(resample_indeces1);  % a nobs x numresample matrix
       simsumm1 = mean(simdata_resampled1,1);  % summary statistics
    elseif  numsimABC>1
       error('at the moment NUMSIMABC can only equal 1.')
    end
    distance = sqrt((simsumm1-sobs).^2);

   % like = sum(1/delta^nsummary * exp(-distance.^2/(2*delta^2)));
   % loglike =  log(like);
    loglike =  -nsummary*log(delta) + logsumexp(-distance.^2/(2*delta^2));
    logprior = -log(hyper_sigma) - 0.5*(mu-hyper_mu)^2 / hyper_sigma^2;
    if log(rand) < loglike-loglike_old + logprior - logprior_old
        accept=accept+1;
         MCMC(mcmc_iter) = mu;
         mu_old = mu;
         loglike_old = loglike;
         logprior_old = logprior;
        % delta = min(delta,prctile(distance,quantile));
    else
         MCMC(mcmc_iter) = mu_old;
    end
end
acceptrate = accept/propose
save('MCMC_resampling.txt','MCMC','-ascii')

figure
plot(MCMC);
titlestring = sprintf('numsimABC = %d, delta = %d',numsimABC,delta);
title(titlestring)

figure
ksdensity(MCMC(burnin:end));
hold on
% plot EXACT POSTERIOR using a conjugate Gaussian prior for mu
posterior_mean = 1/(1/hyper_sigma^2 + nobs/sigma^2) * (hyper_mu/hyper_sigma^2 + sum(data)/sigma^2);
posterior_std = (1/hyper_sigma^2 + nobs/sigma^2)^(-0.5);
x = [-0.15:.0001:0.15];
y = normpdf(x,posterior_mean,posterior_std);
plot(x,y)
%posterior_draws = posterior_mean + posterior_std * randn(10000,1);
%ksdensity(posterior_draws)
legend('ABC with resampling','Exact Bayes')

fprintf('\nABC inference for MU')
[mean(MCMC(burnin:end)), prctile(MCMC(burnin:end),[2.5 97.5])]
fprintf('\nExact inference for MU')
[posterior_mean,posterior_mean-1.96*posterior_std,posterior_mean+1.96*posterior_std]


return

% fprintf('\nABC inference for MU')
% prctile(MCMC(burnin:end),[2.5 50 97.5])
% fprintf('\nExact inference for MU')
% prctile(posterior_draws,[2.5 50 97.5])

% loglikelihood calculations at several parameter values
rng(100)
mu_test = [-0.15:.001:0.15];
numattempts = 100;
loglike_test = zeros(length(mu_test),numattempts);
for attempt = 1:numattempts
   for jj=1:length(mu_test)
      simdatatest = mu_test(jj) + randn(nobs,numsimABC);
      simdata_resampledtest = simdatatest(resample_indeces1);  % a nobs x numresample matrix
      simsummtest = mean(simdata_resampledtest,1);  % summary statistics
      distance = sqrt((simsummtest-sobs).^2);
      loglike_test(jj,attempt) =  -nsummary*log(delta) + logsumexp(-distance.^2/(2*delta^2));
   end
end
plot(mu_test,mean(exp(loglike_test),2),'k-',mu_test,prctile(exp(loglike_test),2.5,2),'g--',mu_test,prctile(exp(loglike_test),97.5,2),'g--')