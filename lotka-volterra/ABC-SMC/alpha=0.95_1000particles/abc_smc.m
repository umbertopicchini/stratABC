function [ABCdraws,final_logweights] = abc_smc(problem,data,bigtheta,sampletime,stoichiometry,parmask,parbase,summ_weights,ABCthreshold,numparticles,alpha)

summobs = feval([problem, '_abc_summaries'],data); % vector of summaries for the observed data

theta = param_mask(bigtheta,parmask); % starting parameter values

nfreepar = length(theta);
ABCdraws = zeros(nfreepar,numparticles);
nsummary = length(summobs); % the number of summary statistics
simsumm_all = zeros(nsummary,numparticles);

% initialization: here t=1
attempts = 0;
while attempts < numparticles
    attempts
        try
          theta = lv_prior([],1);
          bigtheta = param_unmask(theta,parmask,parbase);
          simdata= feval([problem, '_model'],bigtheta,sampletime,stoichiometry,1);
        catch
            try
               theta = lv_prior([],1);
               bigtheta = param_unmask(theta,parmask,parbase);
               simdata= feval([problem, '_model'],bigtheta,sampletime,stoichiometry,1);
            catch
               theta = lv_prior([],1);
               bigtheta = param_unmask(theta,parmask,parbase);
               simdata= feval([problem, '_model'],bigtheta,sampletime,stoichiometry,1);
            end
        end
        if ~any(isnan(simdata))
           attempts = attempts+1;
           simsumm_all(:,attempts) = feval([problem, '_abc_summaries'],simdata);
           ABCdraws(:,attempts) = theta;
        end
end

weights = ones(1,numparticles)/numparticles;
tau = sqrt(2)*std(ABCdraws,0,2);  % Beaumont et al 2009, Biometrika
logweights_old = log(weights);
logweights = zeros(1,numparticles);
ABCthreshold_old = ABCthreshold;
normlogweights_old = (logweights_old-max(logweights_old))-log(sum(exp(logweights_old-max(logweights_old)))); % normalize weights; suggestion from page 6 of Cappe et al. "An overview of existing methods and recent advances in sequential Monte Carlo"
ess_old = 1/sum((exp(normlogweights_old)).^2);  % the Effective Sample Size

numaccept = 0;
numprop = 0;
acceptrate = 1;
t = 1;
save(sprintf('ABCdraws_stage%d.dat',t),'ABCdraws','-ascii')
timesbelowthreshold = 0;
keepsampling = 1;
iter_firsttime_below = 0;
iter_secondtime_below = 0;

while keepsampling && t<1000 % keep doing as long as acceptance rate requirement is satisfied
    t = t+1;
    
    % root finding procedure for the ABC threshold
  %  optimoptions = optimset('Display','iter'); % show iterations
  
    ABCthreshold_old
    optoptions = optimset('Display','iter'); % show iterations
    try
       ABCthreshold = fzero(@(ABCthreshold) abc_findthreshold(ABCthreshold),[0.5*ABCthreshold_old,ABCthreshold_old],optoptions);
    catch
       ABCthreshold = 0.90*ABCthreshold_old; 
    end
    save(sprintf('ABCthreshold_stage%d.dat',t),'ABCthreshold','-ascii')
    xc = bsxfun(@minus,simsumm_all',summobs');
    distance = sqrt(sum((xc / summ_weights) .* xc, 2));
    for ii=1:numparticles 
        loglike_oldthreshold = -nsummary*log(ABCthreshold_old) -distance(ii).^2/(2*ABCthreshold_old^2);
        loglike_newthreshold = -nsummary*log(ABCthreshold) -distance(ii).^2/(2*ABCthreshold^2);
        logweights(ii) = logweights_old(ii) + loglike_newthreshold - loglike_oldthreshold;
    end
    final_logweights = logweights;  % useful to get as output, to possibly compute (weighted) posterior mean, posterior SD
    save(sprintf('logweights_stage%d.dat',t),'final_logweights','-ascii')
    ABCthreshold_old = ABCthreshold;
    normlogweights = (logweights-max(logweights))-log(sum(exp(logweights-max(logweights)))); % normalize weights; suggestion from page 6 of Cappe et al. "An overview of existing methods and recent advances in sequential Monte Carlo"
    ess = 1/sum((exp(normlogweights)).^2);  % the Effective Sample Size
    ess_old = ess;
    save(sprintf('ess_stage%d.dat',t),'ess','-ascii')
    if ess < numparticles/2
        ess
        index = stratresample(exp(normlogweights),numparticles);
        ABCdraws = ABCdraws(:,index);
        simsumm_all = simsumm_all(:,index);
        weights = ones(1,numparticles)./numparticles;
        logweights = log(weights);
        logweights_old = logweights;
        normlogweights_old = (logweights_old-max(logweights_old))-log(sum(exp(logweights_old-max(logweights_old)))); % normalize weights; suggestion from page 6 of Cappe et al. "An overview of existing methods and recent advances in sequential Monte Carlo"
        ess_old = 1/sum((exp(normlogweights_old)).^2);  % the Effective Sample Size
    else
        logweights_old = logweights;
        index = 1:numparticles;
    end
    for ii=1:numparticles
        numprop = numprop+1;
        if exp(logweights(ii))>0
           theta = mvnrnd(ABCdraws(:,ii),tau.^2');
           bigtheta = param_unmask(theta,parmask,parbase);
           simdata = feval([problem, '_model'],bigtheta,sampletime,stoichiometry,1); 
           simsumm = feval([problem, '_abc_summaries'],simdata);
           xc = bsxfun(@minus,simsumm',summobs');
           distance = sqrt(sum((xc / summ_weights) .* xc, 2));
           loglike = -nsummary*log(ABCthreshold) -distance.^2/(2*ABCthreshold^2);
           xc = bsxfun(@minus,simsumm_all(:,ii)',summobs');
           distance = sqrt(sum((xc / summ_weights) .* xc, 2));
           loglike_old = -nsummary*log(ABCthreshold) -distance.^2/(2*ABCthreshold^2);
           prior = lv_prior(theta,0);
           prior_old = lv_prior(ABCdraws(:,ii),0);
           if log(rand) < loglike - loglike_old + log(prior) - log(prior_old)
              ABCdraws(:,ii) = theta;
              simsumm_all(:,ii) = simsumm;
              numaccept = numaccept+1;
           end
        end      
    end
    tau = sqrt(2)*std(ABCdraws,0,2);  % Beaumont et al 2009, Biometrika
    save(sprintf('ABCdraws_stage%d.dat',t),'ABCdraws','-ascii')
    acceptrate = numaccept/numprop
    if acceptrate <= 0.010
         timesbelowthreshold = timesbelowthreshold+1;
         if timesbelowthreshold==1
             iter_firsttime_below = t;
         elseif timesbelowthreshold==2
             iter_secondtime_below = t;
         end
    end
    save(sprintf('acceptrate_stage%d.dat',t),'acceptrate','-ascii')
    numaccept = 0;
    numprop = 0;
    if timesbelowthreshold > 1 
         if iter_secondtime_below == iter_firsttime_below +1 % only stop if two CONSECUTIVE iterations have acceptance rate below the limit
             keepsampling=0;
             %break
         else
             % reset counters since we have reached timesbelowthreshold==2 but
             % not two *consecutive* times when acceptance rate < 1%
             timesbelowthreshold = 0;
             iter_firsttime_below = 0;
             iter_secondtime_below = 0;
         end
    end
end



    function out = abc_findthreshold(ABCthreshold)
           %  ABCthreshold
             xc = bsxfun(@minus,simsumm_all',summobs');
             distance = sqrt(sum((xc / summ_weights) .* xc, 2));
             loglike_oldthreshold = -nsummary*log(ABCthreshold_old) -distance.^2/(2*ABCthreshold_old^2);
             loglike_newthreshold = -nsummary*log(ABCthreshold) -distance.^2/(2*ABCthreshold^2);
         %    size(logweights_old)
         %    size(loglike_newthreshold)
         %    size(loglike_oldthreshold)
             logweights = logweights_old + loglike_newthreshold' - loglike_oldthreshold';
             id_nan = find(isnan(logweights));
             logweights(id_nan) = -inf;
%              normlogweights_old = (logweights_old-max(logweights_old))-log(sum(exp(logweights_old-max(logweights_old))));  % normalize weights; suggestion from page 6 of Cappe et al. "An overview of existing methods and recent advances in sequential Monte Carlo"
%              ess_old = 1/sum((exp(normlogweights_old)).^2);  % the Effective Sample Size
             normlogweights = (logweights-max(logweights))-log(sum(exp(logweights-max(logweights)))); % normalize weights; suggestion from page 6 of Cappe et al. "An overview of existing methods and recent advances in sequential Monte Carlo"
             ess = 1/sum((exp(normlogweights)).^2);  % the Effective Sample Size
             out = ess - alpha * ess_old;  % we want to find the zero of this difference (as a function of ABCthreshold)
    end
end

