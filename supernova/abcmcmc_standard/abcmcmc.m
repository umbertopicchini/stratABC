function ABCMCMC = abcmcmc(problem,summobs,nobs,nbin,bigtheta,parmask,parbase,ABCthreshold,numsim,R_mcmc,burnin,step_rw,length_CoVupdate)

% problem: a string identifying the problem/example at hand
% data: the vector of data (duh!)
% bigtheta: the full vector of free + constant model parameters
% parmask: a 0-1 vector having length equal to length(bigtheta). Has 1 for parameters to estimate, 0 otherwise
% parbase: redundant... this is equal to bigtheta
% covariates: possible vector/matrix/structure of covariates
% R_mcmc_res: Number of iterations for rABC-MCMC
% R_mcmc_strat:  additional number of rsABC-MCMC iterations, following the rABC-MCMC ones
% burnin_abc: the scaling matrix Sigma for the summary statistics is computed on summaries simulated over the initial burnin_abc iterations
% numresample1: number of data resamples used to compute the strata probabilities omega_j
% numresample2: number of data resamples used to compute the number of samples n_j for each stratum
% numsimABC: number of times we simulate from the model at each iteration. 
% step_rw: vector of standard deviations for the Metropolis random walk proposal distribution (on log-scale parameters)
% ABCquantile: the value ofvthe alpha-quantile for the computation of the ABC threshold
% frequency_threshold_upd:  how frequently should we check whether we can reduce the ABC threshold
% kernel: the ABC kernel. Should be 'gauss' (always!) or 'identity' (don't try this!)
% adaptation: the method for adaptive Metropolis. Can be 'am' (recommended) or 'ram' (discouraged). AM is Haario's et al (2001) method. 'ram' is the robust AM by Vihola
% targetrate,gamma,burnin_metropolis: these are settings specific to 'ram', which is anyway doiscouraged
% length_CoVupdate: when adaptation=='am', this is how often we recompute the proposal covariance


%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::
fprintf('\nSimulation has started...')
% extract parameters to be estimated (i.e. those having parmask==1, from the full vector bigtheta)
theta_old = param_mask(bigtheta,parmask); % starting parameter values
ABCMCMC = zeros(R_mcmc,length(theta_old)); 
ABCMCMC(1,:) = theta_old;
nsummary = length(summobs); % the number of summary statistics

covar = eye(nsummary); % initial scaling for the ABC distances

%ABCsummaries = summsimuldata_resampled;

 % centre input
%xc = bsxfun(@minus,ABCsummaries',summobs');
%distance = sqrt(sum((xc / covar) .* xc, 2));  

%abcloglike_old = -nsummary*log(ABCthreshold) +logsumexp(-distance.^2/(2*ABCthreshold^2));

% an arbitrary initial ABC loglikelihood
abcloglike_old = -1e300;

if isnan(abcloglike_old) || isinf(abcloglike_old)
    error('inadmissible initial ABC loglikelihood')
end

% initial (diagonal) covariance matrix for the Gaussian proposal
cov_current = diag(step_rw.^2);
% propose a value for parameters using Gaussian random walk
theta = mvnrnd(theta_old,cov_current);



% reconstruct updated full vector of parameters (i.e. rejoins
% fixed/constant parameters and those to be estimated)
%bigtheta = param_unmask(theta,parmask,parbase);

%simuldata = feval([problem, '_simsummaries'],bigtheta,nobs,nbin); 
% resamples simulated data 
%simuldata_resampled = simuldata(resample_indeces);  
% compute summaries of resampled data
% notice, in this problem data=summaries
%summsimuldata_resampled = simuldata_resampled;

%ABCsummaries = summsimuldata_resampled;

 % centre input
%xc = bsxfun(@minus,ABCsummaries',summobs');
%distance = sqrt(sum((xc / covar) .* xc, 2));  
% define ABC threshold
%abcloglike = -nsummary*log(ABCthreshold) +logsumexp(-distance.^2/(2*ABCthreshold^2));


%collectdistances = [collectdistances;distance];

%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
   % evaluate priors at old parameters
 prior_old =  feval([problem, '_prior'],theta_old);

%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values


for mcmc_iter = 1:R_mcmc
    
    if mcmc_iter == burnin
             if burnin >= length_CoVupdate
                lastCovupdate = burnin-length_CoVupdate;
             end
             cov_last = cov_current;
       end
       if (mcmc_iter < burnin)
          cov_current = diag(step_rw.^2); 
          theta = mvnrnd(theta_old,cov_current);
       else
           if (mcmc_iter == lastCovupdate+length_CoVupdate) 
               covupdate = cov(ABCMCMC(burnin/2:mcmc_iter-1,1:end));
               % compute equation (1) in Haario et al.
               cov_current = (2.38^2)/length(theta)*covupdate +  (2.38^2)/length(theta) * 1e-8 * eye(length(theta)) ;
               theta = mvnrnd(theta_old,cov_current);
               cov_last = cov_current;
               lastCovupdate = mcmc_iter;
               fprintf('\nMCMC iteration -- adapting covariance...')
               fprintf('\nMCMC iteration #%d -- acceptance ratio %4.3f percent',mcmc_iter,accept_proposal/num_proposal*100)
               accept_proposal=0;
               num_proposal=0;
               MCMC_temp = ABCMCMC(1:mcmc_iter-1,:);
               save('THETAmatrix_temp','MCMC_temp');
           else
              % Here there is no "adaptation" for the covariance matrix,
              % hence we use the same one obtained at last update
                theta = mvnrnd(theta_old,cov_last);
           end
       end
     %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::   

    
    
    num_proposal = num_proposal+1;
    bigtheta = param_unmask(theta,parmask,parbase);
    
    simdata= feval([problem, '_simsummaries'],bigtheta,nobs,nbin,numsim); 
  %  simdata2= feval([problem, '_simsummaries'],bigtheta,nobs,nbin);

  %  xc1 = bsxfun(@minus,simdata1,summobs'); 
  %  xc2 = bsxfun(@minus,simdata2,summobs'); 
    xc = bsxfun(@minus,simdata',summobs');
  %  distance1 = sqrt(sum(xc1 .* xc1, 2));
  %  distance2 = sqrt(sum(xc2 .* xc2, 2));
  %  distance = [distance1,distance2];
    distance = sqrt(sum(xc .* xc, 2));
   
   abcloglike =  -nsummary*log(ABCthreshold) +logsumexp(-distance.^2/(2*ABCthreshold^2));
   
   prior = feval([problem, '_prior'],theta);
    
    if log(rand) < abcloglike-abcloglike_old + log(prior) - log(prior_old)
         ABCMCMC(mcmc_iter,:) = theta;
         abcloglike_old = abcloglike;
         theta_old = theta;
         prior_old = prior;
         accept_proposal=accept_proposal+1;
    else
         ABCMCMC(mcmc_iter,:) = theta_old;
    end
end
   
%save('ABCMCMC','ABCMCMC') 
fprintf('\n')


    
end
