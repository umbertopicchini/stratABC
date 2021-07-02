function ABCMCMC = rsabcmcmc(problem,data,bigtheta,simultime,sampletime,stoichiometry,parmask,parbase,numresample1,numresample2,R_mcmc,step_rw,burnin,length_CoVupdate,covar,ABCthreshold)

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
ABCMCMC = zeros(R_mcmc,length(theta_old)); 
ABCMCMC(1,:) = theta_old;


nsummary = length(summobs); % the number of summary statistics
npar = length(theta_old);
nvar = size(data,2);

%resampling indeces, the same across the simulations for numerical
%efficiency
nobs = size(data,1);

simdata_resampled1 = zeros(nobs,numresample1*nvar);
simdata_resampled2 = zeros(nobs,numresample2*nvar);

accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values

% an arbitrary initial ABC loglikelihood
abcloglike_old = -1e300;

prior_old = feval([problem, '_prior'],theta_old);

cov_current = diag(step_rw.^2);



% the main rsABC-MCMC loop
for mcmc_iter = 1:R_mcmc

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
    
   simdata2= feval([problem, '_model'],bigtheta,simultime,sampletime,stoichiometry,1); 
    % resamples simulated data using the overlapping blocks bootstrap with size 8     
    count = 0;
    for jj=1:numresample2
       simdata_resampled2(:,jj+count:jj+1+count) = overlappingBB(simdata2,8);
       count = count +1;
    end
    simsumm2 = feval([problem, '_abc_summaries'],simdata_resampled2);

   xc = bsxfun(@minus,simsumm2',summobs');
   distance = sqrt(sum((xc / covar) .* xc, 2)); 
   index_inclusion1 = distance < 0.5*ABCthreshold;  % was 2.5 and 5
   n1test = sum(index_inclusion1); % number of summaries falling into the ellipsis above
   distance1 = distance(index_inclusion1);
   index_inclusion2 = (distance < 1*ABCthreshold) & ~(distance < 0.5*ABCthreshold);
   n2test = sum(index_inclusion2); % number of summaries falling into the ellipsis above but not in the innermost one 
   distance2 = distance(index_inclusion2);
   index_inclusion3 =  ~(distance < 1*ABCthreshold);
   n3test = sum(index_inclusion3);
   distance3 = distance(index_inclusion3);
    
   if n1test==0 || n2test==0 || n3test==0 % skip the rest of the loop because of neglected strata
      ABCMCMC(mcmc_iter,:) = theta_old;
      continue
   end

    simdata1= feval([problem, '_model'],bigtheta,simultime,sampletime,stoichiometry,1); 
    % resamples simulated data using the overlapping blocks bootstrap with size 8 
     count = 0;
     for jj=1:numresample1
        simdata_resampled1(:,jj+count:jj+1+count) = overlappingBB(simdata1,8);
        count = count +1;
     end
    simsumm1 = feval([problem, '_abc_summaries'],simdata_resampled1);

    xc = bsxfun(@minus,simsumm1',summobs');
    distance = sqrt(sum((xc / covar) .* xc, 2));
    index_inclusion1 = distance < 0.5*ABCthreshold;  % was 2.5 and 5
    n1 = sum(index_inclusion1);
    omega1 = n1/numresample1; 
    index_inclusion2 = (distance < 1*ABCthreshold) & ~(distance < 0.5*ABCthreshold);
    n2 = sum(index_inclusion2);
    omega2 = n2/numresample1;
    omega3 = 1-(omega1+omega2);

   logL1 = log(omega1/n1test) -nsummary*log(ABCthreshold) + logsumexp(-distance1.^2/(2*ABCthreshold^2));
   logL2 = log(omega2/n2test) -nsummary*log(ABCthreshold) + logsumexp(-distance2.^2/(2*ABCthreshold^2));
   logL3 = log(omega3/n3test) -nsummary*log(ABCthreshold) + logsumexp(-distance3.^2/(2*ABCthreshold^2));
   abcloglike =  logsumexp([logL1,logL2,logL3]);
   
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
   

save('ABCMCMC','ABCMCMC') 
save('ABCMCMC.dat','ABCMCMC','-ascii') 
fprintf('\n')


    
end



