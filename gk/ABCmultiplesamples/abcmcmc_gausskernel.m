function ABCMCMC = abcmcmc(problem,data,bigtheta,parmask,parbase,covariates,threshold_vec,updatethreshold,R_mcmc,step_rw,weights,length_CoVupdate,numsimABC)
% an ABC-MCMC algorithm using a Gaussian kernel.
% Output: 
%     - ABCMCMC: an R_mcmc x p matrix containing the R_mcmc draws generated via ABC-MCMC
%          for the p parameters of interest.  I
%     - summariesmatrix: an R_mcmc x d matrix where d is the length of the
%                        vector of summary statistics. This is typically
%                        useful after a first ("pilot") run of abcmcmc is
%                        completed. A possibility is to use some measures of
%                        dispersion (say the square roots of mean absolute deviations) for each
%                        of the d columns of summariesmatrix (after
%                        dicarding some burnin). The user could then construct a
%                        1 x d vector containing such measures of dispersion
%                        and use it as input parameter "weights" for a
%                        second round of abcmcmc.
% Input:
%     - problem: a string indentifying the name of the experiment;
%     - data: a vector of size n containing values of the dependent
%              variable to be modelled.
%     - bigtheta: the full vector of structural model parameters. Contains
%                 both free to vary parameters (unknowns to be estimated)
%                 and constant/known parameters. See also parmask.
%     - parmask:  a vector having the same size as bigtheta. Contains 1's
%                 in correspondence of parameters to be estimated an 0 otherwise
%                 (fixed constants).
%     - parbase:  contains initial values for bigtheta (redundant: it's the same as parbase). 
%     - covariates: additional covariates appearing in the model. Can be
%                   observational times or other variables. 
%     - threshold_vec: the vector of ABC thresholds. Values should be
%                      decreasing.
%     - updatethreshold: a vector of positive integers having size length(threshold_vec)-1. 
%                        Contains the indeces of the iterations where the
%                        threshold should be updated. E.g. if threshold_vec
%                        = [5 3 1] and  updatethreshold = [1000, 2000]
%                        then we wish to use threshold = 5 from iteration 1
%                        to 1000, threshold = 3 from 1001 to 2000,
%                        threshold = 1 from 3000 onward. Typically
%                        max(updatethreshold) << R_mcmc.
%     - R_mcmc:        the total number of draws to be generated.
%     - step_rw :      a vector having length p (where p=number of parameters 
%                      to be estimated) containing the the initial value for
%                      the standard deviations of the multivariate Gaussian proposal
%                      for (adaptive) Metropolis random walk.
%     - weights : a vector of length d (where d is the length of the vector of summaries) containing weights 
%                 to form the diagonal covariance matrix of the summaries to be
%                 used in the multivariate Gaussian ABC kernel.
%     - length_CoVupdate: an integer representing how frequently the covariance
%                 matrix for the proposal function in adaptive ABC-MCMC should be updated (say, every 500 iterations). Because of the
%                 low acceptance rates of ABC procedures it is not advised to update
%                 the covariance nmatrix very frequently.

% Umberto Picchini 2016
% www.maths.lth.se/matstat/staff/umberto/


data=data(:); % assume a column data vector
summobs = feval([problem, '_abc_summaries'],data) % vector of summaries for the observed data
threshold = threshold_vec(1);  % the initial value for the ABC threshold
nsummary = length(summobs);

%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::
% simulate trajectories
simuldata_old = feval([problem, '_modelsimulate'],bigtheta,covariates,numsimABC); 
summsimuldata_old = feval([problem, '_abc_summaries'],simuldata_old);

covar = diag(weights.^2);
% a gaussian kernel with inverse covariance invcovmatrix
xc = bsxfun(@minus,summsimuldata_old',summobs');
distance = sqrt(sum((xc / covar) .* xc, 2));
logkernel_old = -nsummary*log(threshold) +logsumexp(-distance.^2/(2*threshold^2));

fprintf('\nSimulation has started...')
% extract parameters to be estimated (i.e. those having parmask==1, from the full vector bigtheta)
theta_old = param_mask(bigtheta,parmask);
ABCMCMC = zeros(R_mcmc,length(theta_old)); 
ABCMCMC(1,:) = theta_old;
% initial (diagonal) covariance matrix for the Gaussian proposal
cov_current = diag(step_rw.^2);
% propose a value for parameters using Gaussian random walk
theta = mvnrnd(theta_old,cov_current);


% reconstruct updated full vector of parameters (i.e. rejoins
% fixed/constant parameters and those to be estimated)
bigtheta = param_unmask(theta,parmask,parbase);
% simulate a new dataset
simuldata = feval([problem, '_modelsimulate'],bigtheta,covariates,numsimABC); 
summsimuldata = feval([problem, '_abc_summaries'],simuldata);
%logkernel =  log( 1*((summsimuldata(:)-summobs(:))'.*invcovmatrix.*(summsimuldata(:)-summobs(:))'<threshold) );
xc = bsxfun(@minus,summsimuldata',summobs');
distance = sqrt(sum((xc / covar) .* xc, 2));
logkernel = -nsummary*log(threshold) +logsumexp(-distance.^2/(2*threshold^2));

%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
   % evaluate priors at old parameters
   prior_old =  feval([problem, '_prior'],theta_old);

   % evaluate priors at proposed parameters
   prior = feval([problem, '_prior'],theta);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

 if log(rand) < logkernel-logkernel_old +log(prior)-log(prior_old)
      % here we accept our proposal theta
      ABCMCMC(2,:) = [theta];
      logkernel_old = logkernel;
      theta_old = theta;
      prior_old = prior;
 else
     % reject proposal
      ABCMCMC(2,:) = [theta_old];
 end


accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values


for mcmc_iter = 3:R_mcmc
   
   
    % update the value of "threshold" 
      if (ismember(mcmc_iter,updatethreshold)) 
            fprintf('\nABC-MCMC (iter %d). Threshold used = %d',mcmc_iter,threshold)
            [~,location_thresh] = ismember(mcmc_iter,updatethreshold);
            if location_thresh+1 <= length(threshold_vec)
               threshold_new = threshold_vec(location_thresh+1);
            else 
               threshold_new = threshold;
            end
            threshold = threshold_new;
            fprintf('\n*** Now using threshold = %d ***',threshold)
            ABCMCMC_temp = ABCMCMC(1:mcmc_iter-1,:);
            save('ABCMCMC_temp','ABCMCMC_temp')
            clear ABCMCMC_temp
      end
    
    
    %::::::::: ADAPTATION OF THE COVARIANCE MATRIX FOR THE PARAMETERS PROPOSAL :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    %::::::::: here we follow the adaptive Metropolis method as in:
    %::::::::  Haario et al. (2001) "An adaptive Metropolis algorithm", Bernoulli Volume 7, 223-242.


       if mcmc_iter == length_CoVupdate 
             lastCovupdate = 0;
             cov_last = cov_current;
             cov_old = cov_last;
             sum_old = theta_old;
       end
       if (mcmc_iter < length_CoVupdate)
          cov_current = diag(step_rw.^2); 
          theta = mvnrnd(theta_old,cov_current);
       else
           if (mcmc_iter == lastCovupdate+length_CoVupdate) 
               % we do not need to recompute the covariance on the whole
               % past history in a brutal way: we can use a recursive
               % formula. See the reference in the file cov_update.m
               covupdate = cov_update(ABCMCMC(lastCovupdate+1:mcmc_iter-1,1:end),sum_old,length_CoVupdate-1,cov_old);
               sum_old = sum(ABCMCMC(lastCovupdate+1:mcmc_iter-1,1:end));
               cov_old = cov(ABCMCMC(lastCovupdate+1:mcmc_iter-1,1:end));
               % compute equation (1) in Haario et al.
               cov_current = (2.38^2)/length(theta)*covupdate +  (2.38^2)/length(theta) * 1e-6 * eye(length(theta)); 
               theta = mvnrnd(theta_old,cov_current);
               cov_last = cov_current;
               lastCovupdate = mcmc_iter;
               fprintf('\nABC-MCMC iteration -- adapting covariance...')
               fprintf('\nABC-MCMC iteration #%d -- acceptance ratio %4.3f percent',mcmc_iter,accept_proposal/num_proposal*100)
               accept_proposal=0;
               num_proposal=0;
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
    simuldata = feval([problem, '_modelsimulate'],bigtheta,covariates,numsimABC); 
    summsimuldata = feval([problem, '_abc_summaries'],simuldata);
  %  logkernel =  log( 1*((summsimuldata(:)-summobs(:))'.*invcovmatrix.*(summsimuldata(:)-summobs(:))'<threshold) );
    xc = bsxfun(@minus,summsimuldata',summobs');
    distance = sqrt(sum((xc / covar) .* xc, 2));
    logkernel = -nsummary*log(threshold) +logsumexp(-distance.^2/(2*threshold^2));
    % evaluate priors at proposed parameters
    prior = feval([problem, '_prior'],theta);
    
    if log(rand) < logkernel-logkernel_old +log(prior)-log(prior_old)
             accept_proposal=accept_proposal+1;
             ABCMCMC(mcmc_iter,:) = [theta];
             logkernel_old = logkernel;
             theta_old = theta;
             prior_old = prior; 
     else
             ABCMCMC(mcmc_iter,:) = [theta_old];
     end
   
   

end
   
save('ABCMCMC','ABCMCMC') 
fprintf('\n')


    
end



