% an ABC-MCMC algorithm to estimate parameters of a g-and-k distribution.
% See for example Allingham-King-Mengersen 2009, Bayesian
% estimation of quantile distributions, Stat Comput 19.

% Here we use our own custom summary statistics, see gk_summaries.


rng(104545056);  % see the seed for pseudo-random numbers 

% the ground-truth parameters
A=3;    % logA = 1.099
B=1;    % logB = 0
g=2;    % logg = 0.693
k=0.5;  % logk = -0.693

nobs = 2000; % number of observations in the dataset 
problem = 'gk';  % just a string for the problem at hand 

bigtheta_true = [log(A),log(B),log(g),log(k)]; % log-transformed ground truth parameters


% generate data from the model (from g-and-k distributions in this case)
data = feval([problem, '_modelsimulate'],bigtheta_true,nobs,1);
sobs  = gk_abc_summaries(data);  % observed summaries


%::::::::::::::::::::::: end of data generation :::::::::::::::::::::::

%          logA       logB        logg        logk   
parbase = [-log(4)    1            6      log(10)   ]; % starting values for the parameters
parmask = [1          1            1           1    ]; % put 1 for parameters to estimate, 0 otherwise

bigtheta = parbase;


R_mcmc_res = 15000;  % Number of iterations for rABC-MCMC
R_mcmc_strat = 20000; % additional number of rsABC-MCMC iterations, following the rABC-MCMC ones
% the above means that the total number of MCMC iterations is R_mcmc_res + R_mcmc_strat
burnin_abc = 5000; % this means that the scaling matrix Sigma for the summary statistics is computed on summaries simulated over the initial burnin_abc iterations
numresample1 = 500; % number of data resamples used to compute the strata probabilities omega_j
numresample2 = 500;  % number of data resamples used to compute the number of samples n_j for each stratum
numsimABC = 1;  % DO NOT MODIFY. This means that we simulate from the model only once for each iteration. That is resamples are obtained for this single simulation, at every iteration
step_rw = [0.1 0.1 0.1 0.01]; % standard deviations for the Metropolis random walk proposal distribution (on log-scale parameters)
ABCquantile = 5;   % It is the alpha-quantile for the computation of the ABC threshold
frequency_threshold_upd = 500; % how often should we check whether we can reduce the ABC threshold
kernel = 'gauss';  % DO NOT MODIFY 
adaptation = 'am';   % % DO NOT MODIFY 
switch adaptation
    case 'am' % Haario et al, adaptive metropolis
        lengthcovupdate = 500;  % was 100 
        targetrate = [];
        gamma = [];
        burnin_metropolis = [];
    case 'ram' % robust adaptive Metropolis (Vihola)
        targetrate = 0.05;   % was 0.02
        lengthcovupdate = [];
        gamma = 2/3; % must be in the interval (0.5,1]. Vihola uses 2/3
        burnin_metropolis = R_mcmc_res;  % was 10000
end

tic
[stratABCMCMC, ABCthreshold_vec,simsummaries] = qabc_resampling_stratified_3strata_exchanged(problem,data,bigtheta,parmask,parbase,nobs,numresample1,numresample2,numsimABC,ABCquantile,burnin_abc,frequency_threshold_upd,R_mcmc_res,R_mcmc_strat,step_rw,adaptation,targetrate,gamma,burnin_metropolis,lengthcovupdate,kernel); 
eval=toc
save('stratABCMCMC.txt','stratABCMCMC','-ascii')
save('ABCthreshold_vec.txt','ABCthreshold_vec','-ascii')

figure
subplot(2,2,1)
plot(exp(stratABCMCMC(:,1)))
hline(A)
xlabel('A')
subplot(2,2,2)
plot(exp(stratABCMCMC(:,2)))
hline(B)
xlabel('B')
subplot(2,2,3)
plot(exp(stratABCMCMC(:,3)))
hline(g)
xlabel('g')
subplot(2,2,4)
plot(exp(stratABCMCMC(:,4)))
hline(k)
xlabel('k')


