% this demo implements ABC with stratified Monte Carlo with estimated
% probabilities omega and resampling

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
numresample2 = 500;
numrepetitions = 1000;
mu_vec = linspace(-0.1,0.1,50);   % 50 parameter points to try in the interval [-0.1,0.1]. Notice the interval includes the true value mu=0

loglike_vec = zeros(numrepetitions,length(mu_vec));
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

rng(100)  % set to reproduce the same stream of random numbers when computing loglikelihoods

weights = zeros(numrepetitions,3);

for mm = 1:length(mu_vec)
    mm
for rr = 1:numrepetitions
%:::::::::: FIRST LOGLIKELIHOOD ESTIMATION ::::::::::::::::::::::
%rr
%::::: TRAINING SET ::::::::::::::::::::::::::::::::::::::::::::   
simdata1 = mu_vec(mm) + randn(nobs,numsimABC);
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
omega3 = 1-(omega1+omega2);


% :::: TEST SET ::::::::::::::::::::::::::::::::::::::::::::::::::::::
n1 = 0;n2 = 0;n3 = 0;
while n1==0 || n2==0 || n3==0  
simdata2 = mu_vec(mm) + randn(nobs,numsimABC);
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
index_inclusion3 = ~(distance < delta);
n3 = numresample2-n2-n1;
distance3 = distance(index_inclusion3);
end

% compute loglikelihood vis stratified sampling
weights(rr,:) = [omega1/n1,omega2/n2,omega3/n3];
logL1 = log(omega1/n1) -nsummary*log(delta) + logsumexp(-distance1.^2/(2*delta^2));
logL2 = log(omega2/n2) -nsummary*log(delta) + logsumexp(-distance2.^2/(2*delta^2));
logL3 = log(omega3/n3) -nsummary*log(delta) + logsumexp(-distance3.^2/(2*delta^2));

loglike_1 = logsumexp([logL1,logL2,logL3]);

%:::::::::: SECOND LOGLIKELIHOOD ESTIMATION ::::::::::::::::::::::
% HERE WE SWAP SIMDATA2 WITH SIMDATA1, BUT OTHERWISE IT IS THE SAME AS ABOVE
%::::: TRAINING SET :::::::::::::::::::::::::::::::::::::::::::: 
if numsimABC ==1
    % training
    simdata_resampled1 = simdata2(resample_indeces1);  % a nobs x numresample matrix
    simsumm1 = mean(simdata_resampled1,1);  % summary statistics
elseif  numsimABC>1
    error('at the moment NUMSIMABC can only equal 1.')
end
distance = sqrt((simsumm1-sobs).^2);
index_inclusion1 = distance < delta/2;
omega1 = sum(index_inclusion1)/numresample1;
index_inclusion2 = (distance < delta) & ~(distance < delta/2);
omega2 = sum(index_inclusion2)/numresample1; % fraction of training summaries falling into the ellipsis above but not in the innermost one 
omega3 = 1-(omega1+omega2);


% :::: TEST SET ::::::::::::::::::::::::::::::::::::::::::::::::::::::
n1 = 0;n2 = 0;n3 = 0;
while n1==0 || n2==0 || n3==0  
if numsimABC ==1
    % test
    simdata1 = mu_vec(mm) + randn(nobs,numsimABC);
    simdata_resampled2 = simdata1(resample_indeces2);  % a nobs x numresample matrix
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
index_inclusion3 = ~(distance < delta);
n3 = numresample2-n2-n1;
distance3 = distance(index_inclusion3);
end

% compute loglikelihood vis stratified sampling
weights(rr,:) = [omega1/n1,omega2/n2,omega3/n3];
logL1 = log(omega1/n1) -nsummary*log(delta) + logsumexp(-distance1.^2/(2*delta^2));
logL2 = log(omega2/n2) -nsummary*log(delta) + logsumexp(-distance2.^2/(2*delta^2));
logL3 = log(omega3/n3) -nsummary*log(delta) + logsumexp(-distance3.^2/(2*delta^2));

loglike_2 = logsumexp([logL1,logL2,logL3]);

% AVERAGED LIKELIHOOD
loglike_vec(rr,mm) = logsumexp([loglike_1,loglike_2]) -log(2);  % take the sample average of the two stratified likelihoods then compute the logarithm.
                                                                 % That is we return log((L1+L2)/2) = log(L1+L2) - log(2) = log(exp(logL1)+exp(logL2)) -log(2) 
end
end
save('loglike_vec_xrs','loglike_vec')

plot(mu_vec,mean(loglike_vec,1),'y-',mu_vec,prctile(loglike_vec,2.5),'y--',mu_vec,prctile(loglike_vec,97.5),'y--')
