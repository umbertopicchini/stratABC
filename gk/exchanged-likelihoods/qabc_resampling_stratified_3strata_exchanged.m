function [ABCMCMC, ABCthreshold_vec,summsimuldata_final] = qabc(problem,data,bigtheta,parmask,parbase,covariates,numresample1,numresample2,numsimABC,ABCquantile,burnin_abc,frequency_threshold_upd,R_mcmc_res,R_mcmc_strat,step_rw,adaptation,targetrate,gamma,burnin_metropolis,length_CoVupdate,kernel)

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
ABCMCMC = zeros(R_mcmc_res+R_mcmc_strat,length(theta_old)); 
ABCMCMC(1,:) = theta_old;

waitingtoupdate=0;

nsummary = length(summobs); % the number of summary statistics
npar = length(theta_old);

%distance = zeros(numsimABC,1);
ABCthreshold_vec = zeros(R_mcmc_res,1);

% for d=1:numsimABC
%     % sample summaries
%     ABCsummaries = mvnrnd(means',covar);
%     ABCsummaries = ABCsummaries';
%     distance(d) = sqrt((ABCsummaries-summobs)'/covar*(ABCsummaries-summobs));
% end


%resampling indeces, the same across the simulations for numerical
%efficiency
nobs = size(data,1);
resample_indeces = zeros(nobs,numresample2);
for jj=1:numresample2
     indeces = randsample(nobs,nobs,'true');
     resample_indeces(:,jj) = indeces;
%      xhat_subs = xhat(indeces,ii);
%               yobssim_subs = yobssim(indeces,ii);
%               xhat_resampled(:,jj) = xhat_subs;
%               yobssim_resampled(:,jj) = yobssim_subs;
end

% simulate numsimABC artificial datasets 
% *** actually here it HAS TO BE numsimABC=1 ***
simuldata = feval([problem, '_modelsimulate'],bigtheta,covariates,numsimABC); 
simuldata_resampled = zeros(nobs,numresample2);
summsimuldata_resampled = zeros(nsummary,numresample2*numsimABC);
count=0;

if numsimABC ==1  % We always assume this case to hold. The code might not support numsimABC>1
    % resamples simulated data
    simuldata_resampled = simuldata(resample_indeces);  % THIS IS A SUPERFAST OPERATION!!
    % compute summaries of resampled data
    summsimuldata_resampled = feval([problem, '_abc_summaries'],simuldata_resampled);
elseif numsimABC > 1
   for ii=1:numsimABC
       for jj=1:numresample2
          simuldata_resampled(:,jj) = simuldata(resample_indeces(:,jj),ii);
       end
       summsimuldata_resampled(:,count+1:count+numresample2) = feval([problem, '_abc_summaries'],simuldata_resampled);
       count = count + numresample2;
   end
end

covar = eye(nsummary); % initial scaling for the ABC distances

ABCsummaries = summsimuldata_resampled;

collectedsummaries = ABCsummaries';

switch kernel
    case 'identity'
    % centre input
    xc = bsxfun(@minus,ABCsummaries',summobs');
    distance = sqrt(sum((xc / covar) .* xc, 2));
    % define ABC threshold
    ABCthreshold = prctile(distance,ABCquantile);
    abcloglike_old = log(sum( (distance<ABCthreshold) )); % unnormalized loglikelihood
    case 'gauss'
    % centre input
    xc = bsxfun(@minus,ABCsummaries',summobs');
    distance = sqrt(sum((xc / covar) .* xc, 2));  
    % define ABC threshold
    ABCthreshold = prctile(distance,ABCquantile);
    abcloglike_old = -nsummary*log(ABCthreshold) +logsumexp(-distance.^2/(2*ABCthreshold^2));
end

ABCthreshold_vec(1)=ABCthreshold;
collectdistances = distance;

if strcmp(adaptation,'am')
   % initial (diagonal) covariance matrix for the Gaussian proposal
   cov_current = diag(step_rw.^2);
   % propose a value for parameters using Gaussian random walk
   theta = mvnrnd(theta_old,cov_current);
elseif strcmp(adaptation,'ram')
    theta_old = theta_old';  % a column vector
    S = diag(step_rw);
    theta = theta_old + S*randn(npar,1);
    if gamma <=0.5 || gamma>1
        error('gamma must take values in the interval in (0.5,1]')
    end
end


% reconstruct updated full vector of parameters (i.e. rejoins
% fixed/constant parameters and those to be estimated)
bigtheta = param_unmask(theta,parmask,parbase);

% simulate numsimABC artificial datasets
% probably here IT HAS TO BE numsimABC=1
simuldata = feval([problem, '_modelsimulate'],bigtheta,covariates,numsimABC); 
count=0;

if numsimABC ==1
    simuldata_resampled = simuldata(resample_indeces);  % THIS IS A SUPERFAST OPERATION!!
    summsimuldata_resampled = feval([problem, '_abc_summaries'],simuldata_resampled);
elseif numsimABC > 1
   for ii=1:numsimABC
       for jj=1:numresample2
          simuldata_resampled(:,jj) = simuldata(resample_indeces(:,jj),ii);
       end
       summsimuldata_resampled(:,count+1:count+numresample2) = feval([problem, '_abc_summaries'],simuldata_resampled);
       count = count + numresample2;
   end
end

ABCsummaries = summsimuldata_resampled;
collectedsummaries = [collectedsummaries;ABCsummaries'] ;

% centre input
% xc = bsxfun(@minus,ABCsummaries,summobs');
% distance = sqrt(sum((xc / covar) .* xc, 2));
% 
% abcloglike = log(sum(distance<ABCthreshold)); % unnormalized loglikelihood

switch kernel
    case 'identity'
    % centre input
    xc = bsxfun(@minus,ABCsummaries',summobs');
    distance = sqrt(sum((xc / covar) .* xc, 2));
    % define ABC threshold
  %  ABCthreshold = prctile(distance,ABCquantile);
    abcloglike = log(sum( (distance<ABCthreshold) )); % unnormalized loglikelihood
    case 'gauss'
    % centre input
    xc = bsxfun(@minus,ABCsummaries',summobs');
    distance = sqrt(sum((xc / covar) .* xc, 2));  
    % define ABC threshold
  %  ABCthreshold = prctile(sqrt(distance),ABCquantile);
    abcloglike = -nsummary*log(ABCthreshold) +logsumexp(-distance.^2/(2*ABCthreshold^2));
end

collectdistances = [collectdistances;distance];

%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
   % evaluate priors at old parameters
   prior_old =  feval([problem, '_prior'],theta_old);

   % evaluate priors at proposed parameters
   prior = feval([problem, '_prior'],theta);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

 if strcmp(adaptation,'ram')  % Vihola's acceptance probability for robust AM
    acceptprob = min(1,exp(abcloglike-abcloglike_old)*prior/prior_old);
 end

 if log(rand) < abcloglike-abcloglike_old +log(prior)-log(prior_old) 
      % here we accept our proposal theta
      ABCMCMC(2,:) = theta;
      theta_old = theta;
      prior_old = prior;
      ABCthreshold
      if 2 > burnin_abc
          ABCthreshold = min(ABCthreshold,prctile(distance,ABCquantile));
      end
      abcloglike_old = abcloglike;
      % ABCthreshold = min(prctile(distance,ABCquantile),max(0, ABCthreshold-0.2/2*mean(distance)));
      %ABCthreshold = max(0, ABCthreshold-0.2/2*mean(distance));
     % ABCthreshold = prctile(distance,ABCquantile);
      ABCthreshold_vec(2)=ABCthreshold;
 else
     % reject proposal
      ABCMCMC(2,:) = theta_old;
      ABCthreshold_vec(2)=ABCthreshold;
 end

accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values
last_threshold_upd = burnin_abc;


%:::::: START THE rABC-MCMC LOOP FOR R_mcmc_res ITERATIONS :::::::::::::::

for mcmc_iter = 3:R_mcmc_res
  
    if ismember(mcmc_iter,round(R_mcmc_res*[1:5:100]/100))
        fprintf('\nMCMC iter #%d',mcmc_iter)
        fprintf('\nABC-MCMC iteration #%d -- acceptance ratio %4.3f percent',mcmc_iter,accept_proposal/num_proposal*100)
        accept_proposal=0;
        num_proposal=0;
        ABCMCMC_temp = ABCMCMC(1:mcmc_iter-1,:);
        save('ABCMCMC_temp','ABCMCMC_temp')
    end
    %::::::::: ADAPTATION OF THE COVARIANCE MATRIX FOR THE PARAMETERS PROPOSAL :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
   
    switch adaptation
        case 'am'
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
                 sum_old = sum(ABCMCMC(lastCovupdate+1:mcmc_iter-1,:));
                 cov_old = cov(ABCMCMC(lastCovupdate+1:mcmc_iter-1,:));
                 % compute equation (1) in Haario et al.
                 cov_current = (2.38^2)/length(theta)*covupdate +  (2.38^2)/length(theta) * 1e-6 * eye(length(theta)); 
                 theta = mvnrnd(theta_old,cov_current);
                 cov_last = cov_current;
                 lastCovupdate = mcmc_iter;
                 accept_proposal=0;
                 num_proposal=0;
              else
                % Here there is no "adaptation" for the covariance matrix,
                % hence we use the same one obtained at last update
                 theta = mvnrnd(theta_old,cov_last);
              end
          end
        case 'ram'
            % here we use robust adaptive Metropolis
            if mcmc_iter < burnin_metropolis
               [theta,S] = ram(theta_old,mcmc_iter,gamma,S,targetrate,acceptprob);
            else
               theta = theta_old + S*randn(npar,1);
            end
    end

    %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::   

    num_proposal = num_proposal+1;
    % reconstuct full parameters vector using the newly proposed theta
    bigtheta = param_unmask(theta,parmask,parbase);
    
    % simulate numsimABC artificial datasets 
    simuldata = feval([problem, '_modelsimulate'],bigtheta,covariates,numsimABC); 
    count=0;

    if numsimABC ==1
       simuldata_resampled = simuldata(resample_indeces);  % THIS IS A SUPERFAST OPERATION!!
       summsimuldata_resampled = feval([problem, '_abc_summaries'],simuldata_resampled);
    elseif numsimABC > 1
       for ii=1:numsimABC
          for jj=1:numresample2
            simuldata_resampled(:,jj) = simuldata(resample_indeces(:,jj),ii);
          end
           summsimuldata_resampled(:,count+1:count+numresample2) = feval([problem, '_abc_summaries'],simuldata_resampled);
           count = count + numresample2;
       end
    end

    ABCsummaries = summsimuldata_resampled;

    switch kernel
    case 'identity'
      % centre input
      xc = bsxfun(@minus,ABCsummaries',summobs');
      distance = sqrt(sum((xc / covar) .* xc, 2));
      abcloglike = log(sum( (distance<ABCthreshold) )); % unnormalized loglikelihood
    case 'gauss'
      % centre input
      xc = bsxfun(@minus,ABCsummaries',summobs');
      distance = sqrt(sum((xc / covar) .* xc, 2));  
      % define ABC threshold
      % ABCthreshold = prctile(sqrt(distance),ABCquantile);
      abcloglike = -nsummary*log(ABCthreshold) +logsumexp(-distance.^2/(2*ABCthreshold^2));
    end
    
    if mcmc_iter < burnin_abc  % collect ABC summaries, so we can compute a scaling matrix
        collectdistances = [collectdistances;distance];
        collectedsummaries = [collectedsummaries;ABCsummaries'] ;
    elseif mcmc_iter == burnin_abc  % after burnin_abc iterations, compute the scaling matrix for ABC summaries
        mad(collectedsummaries,1,1)
        if sum(mad(collectedsummaries,1,1)==0) > 0
            error('Summaries collected during the ABC burnin phase have a zero median-absolute-deviation. You might want to use the mean-absolute-deviation instead.')
        end
        covar = diag((mad(collectedsummaries,1,1)).^2);  % the scaling matrix
        clear collectedsummaries
    end
    
    
    % evaluate priors at proposed parameters
    prior = feval([problem, '_prior'],theta);
    
    if strcmp(adaptation,'ram')
       acceptprob = min(1,exp(abcloglike-abcloglike_old)*prior/prior_old);
    end
    
    if log(rand) < abcloglike-abcloglike_old +log(prior)-log(prior_old)
             accept_proposal=accept_proposal+1;
             ABCMCMC(mcmc_iter,:) = theta;
             abcloglike_old = abcloglike;
             theta_old = theta;
             prior_old = prior;
             % when a proposal is accepted, we check if we can reduce the
             % ABC threshold. This is checked every frequency_threshold_upd iterations,
             % and it gests reduced only when the sum of distances smaller than the treshold 
             % is at least 2% the number of the proposed summaries
             if mcmc_iter == burnin_abc
                 ABCprevious = ABCthreshold;
                 ABCthreshold = min(ABCthreshold,prctile(collectdistances,ABCquantile));
                 fprintf('\n ABCMCMC iter %d, new ABC threshold is: %d',mcmc_iter,ABCthreshold)
                 last_threshold_upd = mcmc_iter;
             elseif  mcmc_iter > burnin_abc && sum(distance<ABCthreshold) >= 0.05*numsimABC*numresample2 && mcmc_iter == last_threshold_upd + frequency_threshold_upd
                 last_threshold_upd = mcmc_iter;
                 ABCprevious = ABCthreshold;
                 ABCthresholdnew = min(ABCthreshold,prctile(distance,ABCquantile));
                 waitingtoupdate=0;
                 if ABCthresholdnew < ABCthreshold
                     ABCthreshold = ABCthresholdnew;
                     fprintf('\n ABCMCMC iter %d, new ABC threshold is: %d',mcmc_iter,ABCthreshold)
                 elseif ABCthresholdnew==ABCthreshold 
                     ABCthresholdtest = ABCthresholdnew-(prctile(distance,ABCquantile)-ABCthreshold )/5;
                     if ABCthresholdtest > 0
                        ABCthreshold = ABCthresholdtest;
                        fprintf('\n ABCMCMC iter %d, new ABC threshold is: %d',mcmc_iter,ABCthreshold)
                     else
                         ABCthreshold =  ABCthresholdnew;
                     end
                 end
             elseif  mcmc_iter > burnin_abc && (mcmc_iter == last_threshold_upd + frequency_threshold_upd)
                 waitingtoupdate = 1;
             elseif waitingtoupdate && sum(distance<ABCthreshold) >= 0.05*numsimABC*numresample2
                 ABCprevious = ABCthreshold;
                 ABCthresholdnew = min(ABCthreshold,prctile(distance,ABCquantile));
                 waitingtoupdate=0;
                 if ABCthresholdnew < ABCthreshold
                     ABCthreshold = ABCthresholdnew;
                     fprintf('\n ABCMCMC iter %d, new ABC threshold is: %d',mcmc_iter,ABCthreshold)
                     last_threshold_upd = mcmc_iter;
                 elseif ABCthresholdnew==ABCthreshold 
                     ABCthresholdtest = ABCthresholdnew-(prctile(distance,ABCquantile)-ABCthreshold )/5;
                     if ABCthresholdtest > 0
                        ABCthreshold = ABCthresholdtest;
                        last_threshold_upd = mcmc_iter;
                        fprintf('\n ABCMCMC iter %d, new ABC threshold is: %d',mcmc_iter,ABCthreshold)
                     else
                         ABCthreshold =  ABCthresholdnew;
                     end
                 end
             end
             ABCthreshold_vec(mcmc_iter)=ABCthreshold;
             summsimuldata_final = summsimuldata_resampled;
    else  % rejected proposal
             ABCMCMC(mcmc_iter,:) = theta_old;
             if mcmc_iter == burnin_abc
                 waitingtoupdate=1;
             elseif mcmc_iter == last_threshold_upd + frequency_threshold_upd
                 waitingtoupdate=1;
             end
             ABCthreshold_vec(mcmc_iter) = ABCthreshold;
     end

end

%:::::::::: END OF rABC-MCMC ::::::::::::::::::::::::::::::::::::::::::::

%:::::::: START THE STRATIFICATION PROCEDURE ::::::::::::::::::::::::::::

% resampling indices for the TRAINING (to compute the strata probabilities
% omega)
resample_indeces1 = zeros(nobs,numresample1);
for jj=1:numresample1
     indeces = randsample(nobs,nobs,'true');
     resample_indeces1(:,jj) = indeces;
end

% resampling indices for the TESTING (to compute the strata frequencies
% n_j)
resample_indeces2 = zeros(nobs,numresample2);
for jj=1:numresample2
     indeces = randsample(nobs,nobs,'true');
     resample_indeces2(:,jj) = indeces;
end

% % COMPUTE THE FIRST STRATIFIED LIKELIHOOD
% 
% %::::: TRAINING SET ::::::::::::::::::::::::::::::::::::::::::::
% bigtheta_old = param_unmask(theta_old,parmask,parbase);
% omega1=0; omega2=0; omega3=0; % initialize the strata probabilities . Only for the starting iteration of rsABC-MCMC
% while (omega1==0) || (omega2==0) || (omega3==0) % keep simulating until omegas are all positive
%    fprintf('\nInitial probabilities stratification failed. Try again...')
%    simdata1= feval([problem, '_modelsimulate'],bigtheta_old,covariates,numsimABC); 
%    if numsimABC ==1
%       % training
%       simdata_resampled1 = simdata1(resample_indeces1);  % a nobs x numresample matrix
%       simsumm1 = feval([problem, '_abc_summaries'],simdata_resampled1);
%    elseif  numsimABC>1
%       error('at the moment NUMSIMABC can only equal 1.')
%    end
%    xc = bsxfun(@minus,simsumm1',summobs');
%    distance = sqrt(sum((xc / covar) .* xc, 2)); 
%    index_inclusion1 = distance < ABCthreshold/2;
%    n1 = sum(index_inclusion1);
%    omega1 = n1/numresample1; 
%    index_inclusion2 = (distance < ABCthreshold) & ~(distance < ABCthreshold/2);
%    n2 = sum(index_inclusion2);
%    omega2 = n2/numresample1;
%    omega3 = 1-(omega1+omega2);
% end
% 
% % :::: TEST SET ::::::::::::::::::::::::::::::::::::::::::::::::::::::
% n1test=0; n2test=0; n3test=0;
% while (n1test==0) || (n2test==0) || (n3test==0)  % repeat until both frequencies are positive (only for the starting iteration of rsABC-MCMC
%    fprintf('\nInitial frequencies assignment failed. Try again...')
%    simdata2 = feval([problem, '_modelsimulate'],bigtheta_old,covariates,numsimABC);
%    if numsimABC ==1
%        % test
%        simdata_resampled2 = simdata2(resample_indeces2);  % a nobs x numresample matrix
%        simsumm2 = feval([problem, '_abc_summaries'],simdata_resampled2);
%    elseif  numsimABC>1
%        error('at the moment NUMSIMABC can only equal 1.')
%    end
%    xc = bsxfun(@minus,simsumm2',summobs');
%    distance = sqrt(sum((xc / covar) .* xc, 2)); 
%    index_inclusion1 = distance < ABCthreshold/2;
%  %  index_inclusion1 = distance < partition(1);
%    n1test = sum(index_inclusion1); % number of test summaries falling into the ellipsis above
%    distance1 = distance(index_inclusion1);
% %   index_inclusion2 = ~(distance < 1.2*ABCthreshold);
%    index_inclusion2 = (distance < ABCthreshold) & ~(distance < ABCthreshold/2);
%  %  index_inclusion2 =  ~(distance < partition(1));
%    n2test = sum(index_inclusion2); % number of test summaries falling into the ellipsis above but not in the innermost one 
%    distance2 = distance(index_inclusion2);
%    index_inclusion3 =  ~(distance < ABCthreshold);
%    n3test = sum(index_inclusion3);
%    distance3 = distance(index_inclusion3);
% end
% 
% 
% 
% % compute loglikelihood via stratified sampling (two strata). Uses a
% % Gaussian kernel
% logL1_old = log(omega1/n1test) -nsummary*log(ABCthreshold) + logsumexp(-distance1.^2/(2*ABCthreshold^2));
% logL2_old = log(omega2/n2test) -nsummary*log(ABCthreshold) + logsumexp(-distance2.^2/(2*ABCthreshold^2));
% logL3_old = log(omega3/n3test) -nsummary*log(ABCthreshold) + logsumexp(-distance3.^2/(2*ABCthreshold^2));
% abcloglike_old_1 =  logsumexp([logL1_old,logL2_old,logL3_old]); 
% 
% 
% %:::::::::::::: COMPUTE THE SECOND STRATIFIED LIKELIHOOD:::::::::::::::::
% 
% %::::: TRAINING SET ::::::::::::::::::::::::::::::::::::::::::::
% bigtheta_old = param_unmask(theta_old,parmask,parbase);
% omega1=0; omega2=0; omega3=0; % initialize the strata probabilities . Only for the starting iteration of rsABC-MCMC
% while (omega1==0) || (omega2==0) || (omega3==0) % keep simulating until omegas are all positive
%    fprintf('\nInitial probabilities stratification failed. Try again...')
%    simdata1= feval([problem, '_modelsimulate'],bigtheta_old,covariates,numsimABC); 
%    if numsimABC ==1
%       % training
%       simdata_resampled1 = simdata1(resample_indeces1);  % a nobs x numresample matrix
%       simsumm1 = feval([problem, '_abc_summaries'],simdata_resampled1);
%    elseif  numsimABC>1
%       error('at the moment NUMSIMABC can only equal 1.')
%    end
%    xc = bsxfun(@minus,simsumm1',summobs');
%    distance = sqrt(sum((xc / covar) .* xc, 2)); 
%    index_inclusion1 = distance < ABCthreshold/2;
%    n1 = sum(index_inclusion1);
%    omega1 = n1/numresample1; 
%    index_inclusion2 = (distance < ABCthreshold) & ~(distance < ABCthreshold/2);
%    n2 = sum(index_inclusion2);
%    omega2 = n2/numresample1;
%    omega3 = 1-(omega1+omega2);
% end
% 
% % :::: TEST SET ::::::::::::::::::::::::::::::::::::::::::::::::::::::
% n1test=0; n2test=0; n3test=0;
% while (n1test==0) || (n2test==0) || (n3test==0)  % repeat until both frequencies are positive (only for the starting iteration of rsABC-MCMC
%    fprintf('\nInitial frequencies assignment failed. Try again...')
%    simdata2= feval([problem, '_modelsimulate'],bigtheta_old,covariates,numsimABC); 
%    if numsimABC ==1
%        % test
%        simdata_resampled2 = simdata2(resample_indeces2);  % a nobs x numresample matrix
%        simsumm2 = feval([problem, '_abc_summaries'],simdata_resampled2);
%    elseif  numsimABC>1
%        error('at the moment NUMSIMABC can only equal 1.')
%    end
%    xc = bsxfun(@minus,simsumm2',summobs');
%    distance = sqrt(sum((xc / covar) .* xc, 2)); 
%    index_inclusion1 = distance < ABCthreshold/2;
%  %  index_inclusion1 = distance < partition(1);
%    n1test = sum(index_inclusion1); % number of test summaries falling into the ellipsis above
%    distance1 = distance(index_inclusion1);
% %   index_inclusion2 = ~(distance < 1.2*ABCthreshold);
%    index_inclusion2 = (distance < ABCthreshold) & ~(distance < ABCthreshold/2);
%  %  index_inclusion2 =  ~(distance < partition(1));
%    n2test = sum(index_inclusion2); % number of test summaries falling into the ellipsis above but not in the innermost one 
%    distance2 = distance(index_inclusion2);
%    index_inclusion3 =  ~(distance < ABCthreshold);
%    n3test = sum(index_inclusion3);
%    distance3 = distance(index_inclusion3);
% end
% 
% 
% 
% % compute loglikelihood via stratified sampling (two strata). Uses a
% % Gaussian kernel
% logL1_old = log(omega1/n1test) -nsummary*log(ABCthreshold) + logsumexp(-distance1.^2/(2*ABCthreshold^2));
% logL2_old = log(omega2/n2test) -nsummary*log(ABCthreshold) + logsumexp(-distance2.^2/(2*ABCthreshold^2));
% logL3_old = log(omega3/n3test) -nsummary*log(ABCthreshold) + logsumexp(-distance3.^2/(2*ABCthreshold^2));
% abcloglike_old_2 =  logsumexp([logL1_old,logL2_old,logL3_old]); 
% 
% % AVERAGED LIKELIHOOD
% abcloglike_old = logsumexp([abcloglike_old_1,abcloglike_old_2]) -log(2);  % take the sample average of the two stratified likelihoods then compute the logarithm.
%                                                                  % That is we return log((L1+L2)/2) = log(L1+L2) - log(2) = log(exp(logL1)+exp(logL2)) -log(2) 

%*** HARD CODED NEGATIVE LOGLIKELIHOOD!!! ******
% FAR FROM IDEAL SOLUTION, BUT OTHERWISE USING A WHILE LOOP IS VERY
% EXPENSIVE. aND WE SHOULD ZLREADY BE IN A GOOD REGION OF THE SPACE
abcloglike_old = -1e300;

prior_old = feval([problem, '_prior'],theta_old);

% the main rsABC-MCMC loop
for mcmc_iter = R_mcmc_res+1:R_mcmc_res+R_mcmc_strat
    
    if ismember(mcmc_iter,round((R_mcmc_res+R_mcmc_strat)*[1:5:100]/100))
       % fprintf('\nMCMC iter #%d',mcmc_iter)
        fprintf('\nABC-MCMC iteration #%d -- acceptance ratio %4.3f percent',mcmc_iter,accept_proposal/num_proposal*100)
        accept_proposal=0;
        num_proposal=0;
        ABCMCMC_temp = ABCMCMC(1:mcmc_iter-1,:);
        save('ABCMCMC_temp','ABCMCMC_temp')
    end
    
    %::::::::: ADAPTATION OF THE RANDOM WALK PROPOSAL COVARIANCE :::::::::
    if (mcmc_iter == lastCovupdate+length_CoVupdate) 
                 fprintf('\nCovariance gets updated')
                 % we do not need to recompute the covariance on the whole
                 % past history in a brutal way: we can use a recursive
                 % formula. See the reference in the file cov_update.m
                 covupdate = cov_update(ABCMCMC(lastCovupdate+1:mcmc_iter-1,1:end),sum_old,length_CoVupdate-1,cov_old);
                 sum_old = sum(ABCMCMC(lastCovupdate+1:mcmc_iter-1,:));
                 cov_old = cov(ABCMCMC(lastCovupdate+1:mcmc_iter-1,:));
                 % compute equation (1) in Haario et al.
                 cov_current = (2.38^2)/length(theta)*covupdate +  (2.38^2)/length(theta) * 1e-6 * eye(length(theta)); 
                 theta = mvnrnd(theta_old,cov_current);
                 cov_last = cov_current;
                 lastCovupdate = mcmc_iter;
                 accept_proposal=0;
                 num_proposal=0;
    else
                % Here there is no "adaptation" for the covariance matrix,
                % hence we use the same one obtained at last update
                 theta = mvnrnd(theta_old,cov_last);
    end
    
    num_proposal = num_proposal+1;
    bigtheta = param_unmask(theta,parmask,parbase);
    
    %:: COMPUTE THE FIRST LOGLIKELIHOOD :::::::::::::::::::::::::::
    
    %::::: TRAINING SET ::::::::::::::::::::::::::::::::::::::::::::
    simdata1= feval([problem, '_modelsimulate'],bigtheta,covariates,numsimABC); 
    if numsimABC ==1
       % training
       simdata_resampled1 = simdata1(resample_indeces1);  % a nobs x numresample matrix
       simsumm1 = feval([problem, '_abc_summaries'],simdata_resampled1);
    elseif  numsimABC>1
       error('at the moment NUMSIMABC can only equal 1.')
    end
    
    %distance = sqrt((simsumm1-summobs).^2);
    xc = bsxfun(@minus,simsumm1',summobs');
    distance = sqrt(sum((xc / covar) .* xc, 2));
    index_inclusion1 = distance < ABCthreshold/2;
    n1 = sum(index_inclusion1);
    omega1 = n1/numresample1; 
    index_inclusion2 = (distance < ABCthreshold) & ~(distance < ABCthreshold/2);
    n2 = sum(index_inclusion2);
    omega2 = n2/numresample1;
    omega3 = 1-(omega1+omega2);

    % :::: TEST SET ::::::::::::::::::::::::::::::::::::::::::::::::::::::
   simdata2 = feval([problem, '_modelsimulate'],bigtheta,covariates,numsimABC);
   if numsimABC ==1
       % test
       simdata_resampled2 = simdata2(resample_indeces2);  % a nobs x numresample matrix
       simsumm2 = feval([problem, '_abc_summaries'],simdata_resampled2);
   elseif  numsimABC>1
       error('at the moment NUMSIMABC can only equal 1.')
   end
   xc = bsxfun(@minus,simsumm2',summobs');
   distance = sqrt(sum((xc / covar) .* xc, 2)); 
   index_inclusion1 = distance < ABCthreshold/2;
 %  index_inclusion1 = distance < partition(1);
   n1test = sum(index_inclusion1); % number of test summaries falling into the ellipsis above
   distance1 = distance(index_inclusion1);
%   index_inclusion2 = ~(distance < 1.2*ABCthreshold);
   index_inclusion2 = (distance < ABCthreshold) & ~(distance < ABCthreshold/2);
 %  index_inclusion2 =  ~(distance < partition(1));
   n2test = sum(index_inclusion2); % number of test summaries falling into the ellipsis above but not in the innermost one 
   distance2 = distance(index_inclusion2);
   index_inclusion3 =  ~(distance < ABCthreshold);
   n3test = sum(index_inclusion3);
   distance3 = distance(index_inclusion3);

   logL1 = log(omega1/n1test) -nsummary*log(ABCthreshold) + logsumexp(-distance1.^2/(2*ABCthreshold^2));
   logL2 = log(omega2/n2test) -nsummary*log(ABCthreshold) + logsumexp(-distance2.^2/(2*ABCthreshold^2));
   logL3 = log(omega3/n3test) -nsummary*log(ABCthreshold) + logsumexp(-distance3.^2/(2*ABCthreshold^2));
   abcloglike_1 =  logsumexp([logL1,logL2,logL3]); 
   
    %:: COMPUTE THE SECOND LOGLIKELIHOOD :::::::::::::::::::::::::::
    
    %::::: TRAINING SET :::::::::::::::::::::::::::::::::::::::::::: 
    
    %distance = sqrt((simsumm1-summobs).^2);
    xc = bsxfun(@minus,simsumm2',summobs');
    distance = sqrt(sum((xc / covar) .* xc, 2));
    index_inclusion1 = distance < ABCthreshold/2;
    n1 = sum(index_inclusion1);
    omega1 = n1/numresample1; 
    index_inclusion2 = (distance < ABCthreshold) & ~(distance < ABCthreshold/2);
    n2 = sum(index_inclusion2);
    omega2 = n2/numresample1;
    omega3 = 1-(omega1+omega2);

    % :::: TEST SET ::::::::::::::::::::::::::::::::::::::::::::::::::::::

   xc = bsxfun(@minus,simsumm1',summobs');
   distance = sqrt(sum((xc / covar) .* xc, 2)); 
   index_inclusion1 = distance < ABCthreshold/2;
 %  index_inclusion1 = distance < partition(1);
   n1test = sum(index_inclusion1); % number of test summaries falling into the ellipsis above
   distance1 = distance(index_inclusion1);
%   index_inclusion2 = ~(distance < 1.2*ABCthreshold);
   index_inclusion2 = (distance < ABCthreshold) & ~(distance < ABCthreshold/2);
 %  index_inclusion2 =  ~(distance < partition(1));
   n2test = sum(index_inclusion2); % number of test summaries falling into the ellipsis above but not in the innermost one 
   distance2 = distance(index_inclusion2);
   index_inclusion3 =  ~(distance < ABCthreshold);
   n3test = sum(index_inclusion3);
   distance3 = distance(index_inclusion3);

   logL1 = log(omega1/n1test) -nsummary*log(ABCthreshold) + logsumexp(-distance1.^2/(2*ABCthreshold^2));
   logL2 = log(omega2/n2test) -nsummary*log(ABCthreshold) + logsumexp(-distance2.^2/(2*ABCthreshold^2));
   logL3 = log(omega3/n3test) -nsummary*log(ABCthreshold) + logsumexp(-distance3.^2/(2*ABCthreshold^2));
   abcloglike_2 =  logsumexp([logL1,logL2,logL3]); 
   
   
   % AVERAGED LIKELIHOOD
   abcloglike = logsumexp([abcloglike_1,abcloglike_2]) -log(2);  % take the sample average of the two stratified likelihoods then compute the logarithm.
                                                                 % That is we return log((L1+L2)/2) = log(L1+L2) - log(2) = log(exp(logL1)+exp(logL2)) -log(2) 

   
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



