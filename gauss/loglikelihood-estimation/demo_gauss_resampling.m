% this demo impplements ABC with resampling

rng(500)

nobs = 1000;  % MUST BE AN EVEN NUMBER
mu_true = 0;
sigma = 1;
data = mu_true + randn(nobs,sigma); % data are iid Gaussian with mean 0 and variance 1
% asume we wish to make inference for the mean of the observations. Assume
% a conjugate (Gaussian) prior.
hyper_mu = 0.1;     % the prior mean 
hyper_sigma = 0.2;  % the prior standard deviation

sobs = mean(data)
nsummary = length(sobs);

numsimABC = 1;
numresample1 = 500;
numrepetitions = 1000;
%mu_vec = [-1:.1:1];
mu_vec = linspace(-0.1,0.1,50);   % 50 parameter points to try in the interval [-0.1,0.1]. Notice the interval includes the true value mu=0


loglike_vec_rabcmcmc = zeros(numrepetitions,length(mu_vec));

delta = 3e-5;

resample_indeces1 = zeros(nobs,numresample1);
for jj=1:numresample1
     indeces = randsample(nobs,nobs,'true');
     resample_indeces1(:,jj) = indeces;
end

rng(100)  % set to reproduce the same stream of random numbers when computing loglikelihoods

for mm = 1:length(mu_vec)
for rr = 1:numrepetitions

simdata1 = mu_vec(mm) + randn(nobs,numsimABC);
if numsimABC ==1
    % training
    simdata_resampled1 = simdata1(resample_indeces1);  % a nobs x numresample matrix
    simsumm1 = mean(simdata_resampled1,1);  % summary statistics
elseif  numsimABC>1
    error('at the moment NUMSIMABC can only equal 1.')
end
distance = sqrt((simsumm1-sobs).^2);

%like_old = 1/delta^nsummary * sum(exp(-distance.^2/(2*delta^2)));

loglike_vec_rabcmcmc(rr,mm) =  -nsummary*log(delta) -log(numsimABC) + logsumexp(-distance.^2/(2*delta^2));
end
end
save('loglike_vec_rabcmcmc','loglike_vec_rabcmcmc')

plot(mu_vec,mean(loglike_vec_rabcmcmc,1),'b-',mu_vec,prctile(loglike_vec_rabcmcmc,2.5),'b--',mu_vec,prctile(loglike_vec_rabcmcmc,97.5),'b--')
