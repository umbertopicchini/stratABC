% this demo impplements ABC with resampling and stratified Monte Carlo
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
numresample2 = 500;

mu_start = 0;
R_mcmc = 10000;
burnin = 1000;

mu_old = mu_start;
MCMC = zeros(R_mcmc,1);
MCMC(1) = mu_old;
delta = 3e-4;



resample_indeces1 = zeros(nobs,numresample1);
for jj=1:numresample1
     indeces = randsample(nobs,nobs,'true');
     resample_indeces1(:,jj) = indeces;
end

resample_indeces2 = zeros(nobs,numresample2);
for jj=1:numresample2
     indeces = randsample(nobs,nobs,'true');
     resample_indeces2(:,jj) = indeces;
end


%::::: TRAINING SET ::::::::::::::::::::::::::::::::::::::::::::
%omega1 = 0;omega2 = 0;omega3 = 0;
%while omega1==0 || omega2==0 || omega3==0   
simdata1 = mu_old + randn(nobs,numsimABC);
if numsimABC ==1
    % training
    simdata_resampled1 = simdata1(resample_indeces1);  % a nobs x numresample matrix
    simsumm1 = mean(simdata_resampled1,1);  % summary statistics
elseif  numsimABC>1
    error('at the moment NUMSIMABC can only equal 1.')
end
distance = sqrt((simsumm1-sobs).^2);
index_inclusion1 = distance < delta/2;
omega1 = sum(index_inclusion1)/numresample1;
index_inclusion2 = (distance < delta) & ~(distance < delta/2);
omega2 = sum(index_inclusion2)/numresample1; % fraction of training summaries falling into the ellipsis above but not in the innermost one 
%index_inclusion3 = (distance < 4*delta) & ~(distance < 3*delta);
%omega3 = sum(index_inclusion3)/numresample1;
omega3 = 1-(omega1+omega2);
%end

collectedomega = [];
collectedfrequencies = [];

% :::: TEST SET ::::::::::::::::::::::::::::::::::::::::::::::::::::::
n1 = 0;n2 = 0;n3 = 0;
while n1==0 || n2==0 || n3==0  
simdata2 = mu_old + randn(nobs,numsimABC);
if numsimABC ==1
    % test
    simdata_resampled2 = simdata2(resample_indeces2);  % a nobs x numresample matrix
    simsumm2 = mean(simdata_resampled2,1);  % summary statistics
elseif  numsimABC>1
    error('at the moment NUMSIMABC can only equal 1.')
end
distance = sqrt((simsumm2-sobs).^2);
index_inclusion1 = distance < delta/2;
n1 = sum(index_inclusion1); % number of test summaries falling into the ellipsis above
distance1 = distance(index_inclusion1);
index_inclusion2 = (distance < delta) & ~(distance < delta/2);
n2 = sum(index_inclusion2); % number of test summaries falling into the ellipsis above but not in the innermost one 
distance2 = distance(index_inclusion2);
%index_inclusion3 = (distance < 4*delta) & ~(distance < 3*delta);
%n3 = sum(index_inclusion3);
%distance3 = distance(index_inclusion3);
index_inclusion3 = ~(distance < delta);
n3 = numresample2-n2-n1;
distance3 = distance(index_inclusion3);
end

% compute loglikelihood vis stratified sampling
%like_old = omega1/n1 * sum(1/delta^nsummary * exp(-distance1.^2/(2*delta^2))) + omega2/n2 * sum(1/delta^nsummary * exp(-distance2.^2/(2*delta^2))) + omega3/n3 * sum(1/delta^nsummary * exp(-distance3.^2/(2*delta^2))) + omega4/n4 * sum(1/delta^nsummary * exp(-distance4.^2/(2*delta^2)));
%loglike_old =  log(like_old);
logL1_old = log(omega1/n1) -nsummary*log(delta) + logsumexp(-distance1.^2/(2*delta^2));
logL2_old = log(omega2/n2) -nsummary*log(delta) + logsumexp(-distance2.^2/(2*delta^2));
logL3_old = log(omega3/n3) -nsummary*log(delta) + logsumexp(-distance3.^2/(2*delta^2));
loglike_old =  logsumexp([logL1_old,logL2_old,logL3_old]);
logprior_old = -log(hyper_sigma) - 0.5*(mu_old-hyper_mu)^2 / hyper_sigma^2;

ratio_of_likelihoods = [];
standard_abcmcmc_loglike = [];
rsabcmcmc_loglike = [];

accept=0;
propose=0;
strat3_lose = 0;
for mcmc_iter = 2:R_mcmc
    mu = mu_old + 0.1*randn;
    propose=propose+1;
  
    %::::: TRAINING SET ::::::::::::::::::::::::::::::::::::::::::::
    simdata1 = mu + randn(nobs,numsimABC);
    if numsimABC ==1
       % training
       simdata_resampled1 = simdata1(resample_indeces1);  % a nobs x numresample matrix
       simsumm1 = mean(simdata_resampled1,1);  % summary statistics
    elseif  numsimABC>1
       error('at the moment NUMSIMABC can only equal 1.')
    end
    distance = sqrt((simsumm1-sobs).^2);
    index_inclusion1 = distance < delta/2;
    omega1 = sum(index_inclusion1)/numresample1;
    index_inclusion2 = (distance < delta) & ~(distance < delta/2);
    omega2 = sum(index_inclusion2)/numresample1; % fraction of training summaries falling into the ellipsis above but not in the innermost one 
    %index_inclusion3 = (distance < 4*delta) & ~(distance < 3*delta);
    %omega3 = sum(index_inclusion3)/numresample1;
    omega3 = 1-(omega1+omega2);
    

    % :::: TEST SET ::::::::::::::::::::::::::::::::::::::::::::::::::::::
    simdata2 = mu + randn(nobs,numsimABC);
    if numsimABC ==1
       % test
       simdata_resampled2 = simdata2(resample_indeces2);  % a nobs x numresample matrix
       simsumm2 = mean(simdata_resampled2,1);  % summary statistics
    elseif  numsimABC>1
       error('at the moment NUMSIMABC can only equal 1.')
    end
    distance = sqrt((simsumm2-sobs).^2);
    index_inclusion1 = distance < delta/2;
    n1 = sum(index_inclusion1); % number of test summaries falling into the ellipsis above
    distance1 = distance(index_inclusion1);
    index_inclusion2 = (distance < delta) & ~(distance < delta/2);
    n2 = sum(index_inclusion2); % number of test summaries falling into the ellipsis above but not in the innermost one 
    distance2 = distance(index_inclusion2);
    %index_inclusion3 = (distance < 4*delta) & ~(distance < 3*delta);
    %n3 = sum(index_inclusion3); 
    %distance3 = distance(index_inclusion3);
    index_inclusion3 = ~(distance < delta);
    n3 = numresample2-n2-n1;
    distance3 = distance(index_inclusion3);
    
%     if n1==0 || n2==0 || n3==0  || n4==0 
%         mcmc_iter
%         n1
%         n2
%         n3
%         n4
%        error('Some partition of the integration space has zero simulated draws in it.')
%     end
    
  %  like = omega1/n1 * sum(1/delta^nsummary * exp(-distance1.^2/(2*delta^2))) + omega2/n2 * sum(1/delta^nsummary * exp(-distance2.^2/(2*delta^2))) + omega3/n3 * sum(1/delta^nsummary * exp(-distance3.^2/(2*delta^2))) + omega4/n4 * sum(1/delta^nsummary * exp(-distance4.^2/(2*delta^2)));
  %  loglike =  log(like);
  %  loglike =  log(omega1/n1) -nsummary*log(delta) + logsumexp(-distance1.^2/(2*delta^2)) + log(omega2/n2) -nsummary*log(delta) + logsumexp(-distance2.^2/(2*delta^2)) + log(omega3/n3) -nsummary*log(delta) + logsumexp(-distance3.^2/(2*delta^2)) ;
    logL1 = log(omega1/n1) -nsummary*log(delta) + logsumexp(-distance1.^2/(2*delta^2));
    logL2 = log(omega2/n2) -nsummary*log(delta) + logsumexp(-distance2.^2/(2*delta^2));
    logL3 = log(omega3/n3) -nsummary*log(delta) + logsumexp(-distance3.^2/(2*delta^2));
    loglike =  logsumexp([logL1,logL2,logL3]);
    logprior = -log(hyper_sigma) - 0.5*(mu-hyper_mu)^2 / hyper_sigma^2;
    if log(rand) < loglike-loglike_old + logprior - logprior_old
         MCMC(mcmc_iter) = mu;
         accept=accept+1;
         standard_abcmcmc_loglike = [standard_abcmcmc_loglike;-log(numresample2)+logsumexp([logsumexp(-distance1.^2/(2*delta^2)), logsumexp(-distance2.^2/(2*delta^2)), logsumexp(-distance3.^2/(2*delta^2))]) ];
         rsabcmcmc_loglike = [rsabcmcmc_loglike;loglike];
         mu_old = mu;
         loglike_old = loglike;
         logprior_old = logprior;
         collectedomega = [collectedomega; omega1,omega2,omega3];
         collectedfrequencies = [collectedfrequencies;n1,n2,n3];
        % delta = min(delta,prctile(distance,quantile));
%         omega1
%         omega2
%         omega3
%         n1
%         n2
%         n3
%         [omega1/n1,omega2/n2,omega3/n3]
          
    else
         MCMC(mcmc_iter) = mu_old;
    end
    minweight = min([omega1/n1,omega2/n2,omega3/n3]);
    if omega3/n3 == minweight
       strat3_lose = strat3_lose+1;
    end
end
acceptrate = accept/propose
strat3_lose/propose
save('MCMC_3strata.txt','MCMC','-ascii')
collectedweights = collectedomega./collectedfrequencies;
figure
subplot(2,2,1)
hist(collectedweights(:,1),20)
subplot(2,2,2)
hist(collectedweights(:,2),20)
subplot(2,2,3)
hist(collectedweights(:,3),20)



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
% posterior_draws = posterior_mean + posterior_std * randn(10000,1);
% ksdensity(posterior_draws)
legend('ABC with resampling + stratification','Exact Bayes')

fprintf('\nABC inference for MU')
[mean(MCMC(burnin:end)), prctile(MCMC(burnin:end),[2.5 97.5])]
fprintf('\nExact inference for MU')
[posterior_mean,posterior_mean-1.96*posterior_std,posterior_mean+1.96*posterior_std]

% fprintf('\nABC inference for MU')
% prctile(MCMC(burnin:end),[2.5 50 97.5])
% fprintf('\nExact inference for MU')
% prctile(posterior_draws,[2.5 50 97.5])




return

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
      % training
      index_inclusion1 = distance < delta;
      omega1 = sum(index_inclusion1)/numresample1;
   %   index_inclusion2 = ~(distance < delta);
   %   omega2 = sum(index_inclusion2)/numresample1; % fraction of training summaries falling into the ellipsis above but not in the innermost one 
   %   index_inclusion3 = (distance < 1.5*delta) & ~(distance < delta);
   %   omega3 = sum(index_inclusion3)/numresample1;
      omega2 = 1-(omega1);
      % test
      simdatatest2 = mu + randn(nobs,numsimABC);
      simdata_resampledtest2 = simdata2(resample_indeces2);  % a nobs x numresample matrix
      simsummtest2 = mean(simdata_resampledtest2,1);  % summary statistics
      distance = sqrt((simsummtest2-sobs).^2);
      index_inclusion1 = distance < delta;
      n1 = sum(index_inclusion1); % number of test summaries falling into the ellipsis above
      distance1 = distance(index_inclusion1);
      index_inclusion2 = ~(distance < delta);
      n2 = sum(index_inclusion2); % number of test summaries falling into the ellipsis above but not in the innermost one 
      distance2 = distance(index_inclusion2);
 %     index_inclusion3 = (distance < 1.5*delta) & ~(distance < delta);
 %     n3 = sum(index_inclusion3);
 %     distance3 = distance(index_inclusion3);
 %     index_inclusion4 = ~(distance < 1.5*delta);
 %     n4 = numresample2-n3-n2-n1;
 %     distance4 = distance(index_inclusion4);
 %     loglike_test(jj,attempt) =  log(omega1/n1) -nsummary*log(delta) + logsumexp(-distance1.^2/(2*delta^2)) + log(omega2/n2) -nsummary*log(delta) + logsumexp(-distance2.^2/(2*delta^2)) + log(omega3/n3) -nsummary*log(delta) + logsumexp(-distance3.^2/(2*delta^2)) +  log(omega4/n4) -nsummary*log(delta) + logsumexp(-distance4.^2/(2*delta^2));
       loglike_test(jj,attempt) =  log(omega1/n1) -nsummary*log(delta) + logsumexp(-distance1.^2/(2*delta^2)) + log(omega2/n2) -nsummary*log(delta) + logsumexp(-distance2.^2/(2*delta^2));
   end
end
plot(mu_test,mean(exp(loglike_test),2),'k-',mu_test,prctile(exp(loglike_test),2.5,2),'g--',mu_test,prctile(exp(loglike_test),97.5,2),'g--')
