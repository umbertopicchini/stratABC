function MCMC = abcmcmc(problem,data,bigtheta,sampletime,stoichiometry,parmask,parbase,R_mcmc,step_rw,burnin,length_CoVupdate,covar,ABCthreshold,numsim)

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

summobs = feval([problem, '_abc_summaries'],data); % vector of summaries for the observed data

%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::
fprintf('\nSimulation has started...')
% extract parameters to be estimated (i.e. those having parmask==1, from the full vector bigtheta)
theta_old = param_mask(bigtheta,parmask); % starting parameter values
MCMC = zeros(R_mcmc,length(theta_old)); 
MCMC(1,:) = theta_old;


nsummary = length(summobs); % the number of summary statistics

% an arbitrary initial ABC loglikelihood
abcloglike_old = -1e300;

prior_old = feval([problem, '_prior'],theta_old);

cov_current = diag(step_rw.^2);

accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values


% the main rsABC-MCMC loop
for mcmc_iter = 2:R_mcmc
 %   mcmc_iter
    %::::::::: ADAPTATION OF THE COVARIANCE MATRIX FOR THE PARAMETERS PROPOSAL :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    %::::::::: here we follow the adaptive Metropolis method as in:
    %::::::::  Haario et al. (2001) "An adaptive Metropolis algorithm", Bernoulli Volume 7, 223-242.

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
               covupdate = cov(MCMC(burnin/2:mcmc_iter-1,1:end));
               % compute equation (1) in Haario et al.
               cov_current = (2.38^2)/length(theta)*covupdate +  (2.38^2)/length(theta) * 1e-8 * eye(length(theta)) ;
               theta = mvnrnd(theta_old,cov_current);
               cov_last = cov_current;
               lastCovupdate = mcmc_iter;
               fprintf('\nMCMC iteration -- adapting covariance...')
               fprintf('\nMCMC iteration #%d -- acceptance ratio %4.3f percent',mcmc_iter,accept_proposal/num_proposal*100)
               accept_proposal=0;
               num_proposal=0;
               MCMC_temp = MCMC(1:mcmc_iter-1,:);
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
    simsumm = zeros(9,numsim);
    for ii=1:numsim
       simdata= feval([problem, '_model'],bigtheta,sampletime,stoichiometry,1);
       simulated_summ = feval([problem, '_abc_summaries'],simdata);
       simsumm(:,ii) = simulated_summ;
    end

    %distance = sqrt((simsumm1-summobs).^2);
    xc = bsxfun(@minus,simsumm',summobs');
    distance = sqrt(sum((xc / covar) .* xc, 2));

    abcloglike = -nsummary*log(ABCthreshold) + logsumexp(-distance.^2/(2*ABCthreshold^2));
    prior = feval([problem, '_prior'],theta);
    
    if log(rand) < abcloglike-abcloglike_old + log(prior) - log(prior_old)
         MCMC(mcmc_iter,:) = theta;
         abcloglike_old = abcloglike;
         theta_old = theta;
         prior_old = prior;
         accept_proposal=accept_proposal+1;
    else
         MCMC(mcmc_iter,:) = theta_old;
    end
    
end
   

save('MCMC','MCMC') 
fprintf('\n')


    
end



