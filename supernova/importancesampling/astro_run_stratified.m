
rng(1234)  % was rng(1234)

% ground-truth parameters
om_true = 0.3;
ok_true = 0;
w0_true = -1;
wa_true = 0;
h0_true = 0.7;

logom_true = log(om_true); % -0.52
logh0_true = log(h0_true); % -0.357
bigtheta_true = [logom_true,    ok_true,     w0_true,  wa_true,  logh0_true];  % ground-truth parameters
parbase =       [     -0.11            0        -0.5         0   logh0_true];

nobs = 10000;  % data sample size
nbin = 20;


% simulate "observed" summaries
s_obs = astro_simsummaries(bigtheta_true,nobs,nbin);
nsummary = length(s_obs);
%:::: end of data generation :::::::::::::::::::::::::::::::::::::::::::



problem = 'astro';
numpostsamples=1000;  % number of desired posterior samples
numresample1 = 3000; % number of data resamples used to compute the strata probabilities omega_j
numresample2 = numresample1;  % number of data resamples used to compute the number of samples n_j for each stratum
M = 2;

resample_indeces1 = zeros(nsummary,numresample1);
for jj=1:numresample1
     indeces = randsample(nsummary,nsummary,'true');
     resample_indeces1(:,jj) = indeces;
end

resample_indeces2 = zeros(nsummary,numresample2);
for jj=1:numresample2
     indeces = randsample(nsummary,nsummary,'true');
     resample_indeces2(:,jj) = indeces;
end

%::::::::::: PILOT TO FIND QUANTILES

% simulate distances however not from the prior predictive.
% we propose from the importance sampler instead
numattempts = 2000;
pilot_numresample = numresample2;  % just for the purpose of producing some distances on resampled data
% resample_indeces = zeros(length(s_obs),pilot_numresample);
% for jj=1:pilot_numresample
%      indeces = randsample(length(s_obs),length(s_obs),'true');
%      resample_indeces(:,jj) = indeces;
% end
alldistances = [];
for ii=1:numattempts
     mean_om_is = 0.327;
     std_om_is = sqrt(2)*sqrt(0.0125);
     om_is = mytruncgaussrandraw([0,1,mean_om_is,std_om_is],rand);
     theta(1) = log(om_is);
     mean_w0_is = -1.002;
     std_w0_is = sqrt(2)*sqrt(0.279);
     w0_is = normrnd(mean_w0_is ,std_w0_is);
     theta(2) = w0_is;
     bigtheta = [theta(1),ok_true,theta(2),wa_true,logh0_true];
     simsumm = astro_simsummaries(bigtheta,nobs,nbin);
     simsumm_resampled = simsumm(resample_indeces2); 
     % we have to sort the output, because the astro model uses centres of
    % histogram bins (which arte of course sorted) as output. In fact summobs
    % is ordered.
     simsumm_resampled = sort(simsumm_resampled);
     xc = bsxfun(@minus,simsumm_resampled',s_obs') ;
   %  sqrt(sum(xc .* xc, 2))
     alldistances = [alldistances;sqrt(sum(xc .* xc, 2))]; 
end
save('alldistances.dat','alldistances','-ascii')
histogram(alldistances)
prctile(alldistances,[0.01:0.01:0.5])
return

%::::::: INFERENCE ::::::::::::::::::::::::::::::::::::::::::::::::::
rng(1234)

ABCthreshold = 5*0.15;

bigtheta =  parbase;

logweights = zeros(1,numpostsamples);
postsamples = zeros(numpostsamples,2);
lognormpdf = @(x,m,s)(-0.5*log(2*pi)-log(s)-(x-m)^2 / (2*s^2));
totalattempts=0;
tStart = tic;
for ii=1:numpostsamples
    ii
       mean_om_is = 0.327;
       std_om_is = sqrt(2)*sqrt(0.0125);
       mean_w0_is = -1.002;
       std_w0_is = sqrt(2)*sqrt(0.279);
       reject=1;
     while reject
       totalattempts=totalattempts+1;
       om_is = mytruncgaussrandraw([0,1,mean_om_is,std_om_is],rand);
       theta(1) = log(om_is);
       w0_is = normrnd(mean_w0_is ,std_w0_is);
       theta(2) = w0_is;
       bigtheta(1) = theta(1);  % the model simulator wants logom not om
       bigtheta(3)= theta(2);

       s2 = astro_simsummaries(bigtheta,nobs,nbin);
       simdata_resampled2 = s2(resample_indeces2);
       simdata_resampled2 = sort(simdata_resampled2);
       % notice, in this problem data=summaries
       simsumm2 = simdata_resampled2';
       xc = bsxfun(@minus,simsumm2,s_obs');
       distance = sqrt(sum(xc .* xc, 2));
       index_inclusion1 = distance < 0.29;
       n1test = sum(index_inclusion1); % number of summaries falling into the ellipsis above
       distance1 = distance(index_inclusion1);
       index_inclusion2 = (distance < 0.42) & ~(distance < 0.29); 
%       index_inclusion2 = ~(distance < 0.30);
       n2test = sum(index_inclusion2); % number of summaries falling into the ellipsis above but not in the innermost one 
       distance2 = distance(index_inclusion2);
       index_inclusion3 =  ~(distance < 0.42);
       n3test = sum(index_inclusion3);
       distance3 = distance(index_inclusion3);
       if n1test==0 || n2test==0 || n3test==0  % skip the rest of the while loop because of neglected strata
           continue
       end

       s1 = astro_simsummaries(bigtheta,nobs,nbin);
       simdata_resampled1 = s1(resample_indeces1);
       simdata_resampled1 = sort(simdata_resampled1);
       % notice, in this problem data=summaries
       simsumm1 = simdata_resampled1';
       xc = bsxfun(@minus,simsumm1,s_obs');
       distance = sqrt(sum(xc .* xc, 2));
       index_inclusion1 = distance < 0.29;
       n1 = sum(index_inclusion1);
       omega1 = n1/numresample1; 
       index_inclusion2 = (distance < 0.42) & ~(distance < 0.29);
       n2 = sum(index_inclusion2);
       omega2 = n2/numresample1;
       omega3 = 1-(omega1+omega2);
       
       logL1 = log(omega1/n1test) -nsummary*log(ABCthreshold) + logsumexp(-distance1.^2/(2*ABCthreshold^2));
       logL2 = log(omega2/n2test) -nsummary*log(ABCthreshold) + logsumexp(-distance2.^2/(2*ABCthreshold^2));
       logL3 = log(omega3/n3test) -nsummary*log(ABCthreshold) + logsumexp(-distance3.^2/(2*ABCthreshold^2));
       abclogkernel =  logsumexp([logL1,logL2,logL3]);
  %     abclogkernel =  logsumexp([logL1,logL2]);
       
       logKernel_0 = -nsummary*log(ABCthreshold);
       if log(rand) <  abclogkernel - logKernel_0 
           logweights(ii) = log(astro_prior(theta)) -lognormpdf(om_is,mean_om_is,std_om_is)-lognormpdf(w0_is,mean_w0_is,std_w0_is);
           postsamples(ii,:) = [exp(theta(1)),theta(2)];
           break
       end
     end
end
eval_time = toc(tStart)
filename = sprintf('strat_evaltime');
save(filename,'eval_time','-ascii')
totalattempts
filename = sprintf('totalattempts_threshold=%g_M=%g',ABCthreshold,M);
save(filename,'totalattempts','-ascii')
weights = exp(logweights);
weights_and_samples = [weights',postsamples];
filename = sprintf('strat_weights_and_samples_threshold=%g_M=%g',ABCthreshold,M);
save(filename,'weights_and_samples','-ascii')
normweights = weights / sum(weights); % normalised weights, to compute the ESS
ESS = 1/(sum(normweights.^2))
filename = sprintf('strat_ESS_threshold=%g_M=%g',ABCthreshold,M);
save(filename,'ESS','-ascii')
% we now want to find a credible interval. 
% to find posterior quantiles from importance sampling we use equation
% (3.6) from Chen & Chao "Monte Carlo Estimation of Bayesian Credible and
% HPD Intervals", JCGS 1999.
% We start with om:
[om_sorted,indeces] = sort(postsamples(:,1));
id_lower = find(cumsum(normweights(indeces))<0.025);
id_lower = id_lower(end)+1;
om_lower_quantile = om_sorted(id_lower);
id_upper = find(cumsum(normweights(indeces))<0.975);
id_upper = id_upper(end)+1;
om_upper_quantile = om_sorted(id_upper);
cred_interval_om = [om_lower_quantile,om_upper_quantile]
post_mean_om = sum(normweights.*postsamples(:,1)') % the posterior mean
% let's do w0 now
[w0_sorted,indeces] = sort(postsamples(:,2));
id_lower = find(cumsum(normweights(indeces))<0.025);
id_lower = id_lower(end)+1;
w0_lower_quantile = w0_sorted(id_lower);
id_upper = find(cumsum(normweights(indeces))<0.975);
id_upper = id_upper(end)+1;
w0_upper_quantile = w0_sorted(id_upper);
cred_interval_w0 = [w0_lower_quantile,w0_upper_quantile]
post_mean_w0 = sum(normweights.*postsamples(:,2)') % the posterior mean

