% an ABC-MCMC algorithm to estimate parameters of a g-and-k distribution.
% See for example Allingham-King-Mengersen 2009, Bayesian
% estimation of quantile distributions, Stat Comput 19.

% Here we use our own custom summary statistics, see gk_summaries.

% Umberto Picchini 2016
% www.maths.lth.se/matstat/staff/umberto/

rng(104545056); 

% the ground-truth parameters
A=3;
B=1;
g=2;
k=0.5;

nobs = 2000; % number of observations in the dataset 
problem = 'gk';  % just a string for the problem at hand 
numattempts = 1;

bigtheta_true = [log(A),log(B),log(g),log(k)]; % the ground truth parameters


% generate data from the model (from g-and-k distributions in this case)
data = feval([problem, '_modelsimulate'],bigtheta_true,nobs,1);

%::::::::::::::::::::::: end of data generation :::::::::::::::::::::::

%          logA       logB        logg        logk   
parbase = [log(A)   log(B)        log(g)     log(k)]; % starting values for the parameters
parmask = [1            1             1         1  ]; % put 1 for parameters to estimate, 0 otherwise

bigtheta = parbase;

%::: PILOT ABC settings ::::::::::::::::::::::::::::::::::::::::::::::::::
updatethreshold = [1500 3500 6000]; % iterations where we want the ABC treshold to be decreased
numsimABC = 500;
threshold_vec = [4 1 0.2 0.0264]; % values of the ABC thresholds for gaussian kernel
R_mcmc = 20000; % number of ABC-MCMC iterations
step_rw = [0.1 0.1 0.1 0.1];  % starting standard deviations for adaptive Metropolis proposal
weights = [7.2284  42.9429 0.0717  7.0104];  % summary statistics weights for a first pilot run of abcmcmc. For further runs read below.
lengthCovUpdate = 500; % how often we update the covariance for the parameters proposal function (adaptive Gaussian random walk)
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if R_mcmc <= max(updatethreshold)
    error('MCMC iterations R_mcmc should be larger than the last element in updatethreshold.')
end

% an intial pilot useful to determine weights to normalize the summary statistics
ABCMCMC_pilot = abcmcmc_gausskernel(problem,data,bigtheta,parmask,parbase,nobs,threshold_vec,updatethreshold,R_mcmc,step_rw,weights,lengthCovUpdate,numsimABC);

figure
subplot(2,2,1)
plot(exp(ABCMCMC_pilot(:,1)))
hline(A)
subplot(2,2,2)
plot(exp(ABCMCMC_pilot(:,2)))
hline(B)
subplot(2,2,3)
plot(exp(ABCMCMC_pilot(:,3)))
hline(g)
subplot(2,2,4)
plot(exp(ABCMCMC_pilot(:,4)))
hline(k)

[mean(exp(ABCMCMC_pilot(8000:end,1))), prctile(exp(ABCMCMC_pilot(8000:end,1)),[2.5 97.5])]
[mean(exp(ABCMCMC_pilot(8000:end,2))), prctile(exp(ABCMCMC_pilot(8000:end,2)),[2.5 97.5])]
[mean(exp(ABCMCMC_pilot(8000:end,3))), prctile(exp(ABCMCMC_pilot(8000:end,3)),[2.5 97.5])]
[mean(exp(ABCMCMC_pilot(8000:end,4))), prctile(exp(ABCMCMC_pilot(8000:end,4)),[2.5 97.5])]





