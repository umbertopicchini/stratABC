
rng(1234)  % was rng(1234)

% ground-truth parameters
om_true = 0.3;
ok_true = 0;
w0_true = -1;
wa_true = 0;
h0_true = 0.7;

logom_true = log(om_true); % -0.52
logh0_true = log(h0_true); % -0.357



nobs = 10000;  % data sample size
nbin = 20;

bigtheta_true = [logom_true,    ok_true,     w0_true,  wa_true,  logh0_true];  % ground-truth parameters
parbase =       [  -0.11               0        -0.5         0        logh0_true];
parmask =       [   1                  0           1         0          0  ];

% simulate "observed" summaries
s_obs = astro_simsummaries(bigtheta_true,nobs,nbin);
nsummary = length(s_obs);
%:::: end of data generation :::::::::::::::::::::::::::::::::::::::::::


problem = 'astro';
numresample1 = 3000; % number of data resamples used to compute the strata probabilities omega_j
numresample2 = numresample1;  % number of data resamples used to compute the number of samples n_j for each stratum
M = 2;

%::::::::::: PILOT TO FIND QUANTILES

% simulate distances from the prior predictive.

numattempts = 2000;
prior_pred_numresample = numresample1;  % just for the purpose of producing some distances on resampled data
resample_indeces = zeros(length(s_obs),prior_pred_numresample);
for jj=1:prior_pred_numresample
     indeces = randsample(length(s_obs),length(s_obs),'true');
     resample_indeces(:,jj) = indeces;
end
alldistances = [];
for ii=1:numattempts
     om_prior      = betarnd(3,3);
     w0_prior      = normrnd(-0.5,0.5);
     logom_prior = log(om_prior);
     theta = [logom_prior,w0_prior];
     bigtheta = [theta(1),ok_true,theta(2),wa_true,logh0_true];
     simsumm = astro_simsummaries(bigtheta,nobs,nbin);
     simsumm_resampled = simsumm(resample_indeces); 
     % we have to sort the output, because the astro model uses centres of
    % histogram bins (which arte of course sorted) as output. In fact summobs
    % is ordered.
     simsumm_resampled = sort(simsumm_resampled);
     xc = bsxfun(@minus,simsumm_resampled',s_obs') ;
   %  sqrt(sum(xc .* xc, 2))
     alldistances = [alldistances;sqrt(sum(xc .* xc, 2))]; 
end

histogram(alldistances)
prctile(alldistances,[0.1:0.1:0.5])
save('pilot_all_distances','alldistances','-ascii')
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
return

% resampling indices (to compute the strata probabilities
% omega)
resample_indeces1 = zeros(nsummary,numresample1);
for jj=1:numresample1
     indeces = randsample(nsummary,nsummary,'true');
     resample_indeces1(:,jj) = indeces;
end

% resampling indices (to compute the strata frequencies
% n_j)
resample_indeces2 = zeros(nsummary,numresample2);
for jj=1:numresample2
     indeces = randsample(nsummary,nsummary,'true');
     resample_indeces2(:,jj) = indeces;
end


% PILOT TO FIND "C", THE NORMALIZING CONSTANT (see algorithm 1 in Bornn, L., Pillai, N. S., Smith, A., & Woodard, D. (2017). The use of a single pseudo-sample in approximate Bayesian computation. Statistics and Computing, 27(3), 583-590.) 
%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
bigtheta =  parbase;
ABCthreshold =  0.15;
size_pilot = 1000;
all_pilot_loglik = zeros(size_pilot,1);
for ii=1:size_pilot
       theta = astro_prior_sample;
       bigtheta(1) = log(theta(1));  % the model simulator wants logom not om
       bigtheta(3) = theta(2);
       
       s2 = astro_simsummaries(bigtheta,nobs,nbin);
       simdata_resampled2 = s2(resample_indeces2);
       simdata_resampled2 = sort(simdata_resampled2);
       % notice, in this problem data=summaries
       simsumm2 = simdata_resampled2';
       xc = bsxfun(@minus,simsumm2,s_obs');
       distance = sqrt(sum(xc .* xc, 2));
       index_inclusion1 = distance < 0.45;
       n1test = sum(index_inclusion1); % number of test summaries falling into the ellipsis above
       distance1 = distance(index_inclusion1);
       index_inclusion2 = (distance < 0.55) & ~(distance < 0.45);
       n2test = sum(index_inclusion2); % number of test summaries falling into the ellipsis above but not in the innermost one 
       distance2 = distance(index_inclusion2);
       index_inclusion3 =  ~(distance < 0.55);
       n3test = sum(index_inclusion3);
       distance3 = distance(index_inclusion3);
       if n1test==0 || n2test==0 || n3test==0 % skip the rest of the loop because of neglected strata
           continue
       end

       s1 = astro_simsummaries(bigtheta,nobs,nbin);
       simdata_resampled1 = s1(resample_indeces1);
       simdata_resampled1 = sort(simdata_resampled1);
       % notice, in this problem data=summaries
       simsumm1 = simdata_resampled1';
       xc = bsxfun(@minus,simsumm1,s_obs');
       distance = sqrt(sum(xc .* xc, 2));
       index_inclusion1 = distance < 0.45;  % was 2.1
       n1 = sum(index_inclusion1);
       omega1 = n1/numresample1; 
       index_inclusion2 = (distance < 0.55) & ~(distance < 0.45); % was 2.1 and 2.4
       n2 = sum(index_inclusion2);
       omega2 = n2/numresample1;
       omega3 = 1-(omega1+omega2);

       logL1 = log(omega1/n1test) -nsummary*log(ABCthreshold) + logsumexp(-distance1.^2/(2*ABCthreshold^2));
       logL2 = log(omega2/n2test) -nsummary*log(ABCthreshold) + logsumexp(-distance2.^2/(2*ABCthreshold^2));
       logL3 = log(omega3/n3test) -nsummary*log(ABCthreshold) + logsumexp(-distance3.^2/(2*ABCthreshold^2));
       abclogkernel =  logsumexp([logL1,logL2,logL3]);
       all_pilot_loglik(ii) = abclogkernel;
end

sum(isnan(all_pilot_loglik))  
histogram(exp(all_pilot_loglik))
return

%::::::: END OF THE PILOT :::::::::::::::::::::::::::::::::::::::::::

% we found that we can take c= max(exp(all_pilot_loglik)) as normalizing constant for the kernel 

%::::::: INFERENCE ::::::::::::::::::::::::::::::::::::::::::::::::::
rng(1234)
c_norm_const = 12e13;
bigtheta =  parbase;
reject = 1;
totalattempts = 0;
numpostsamples=1000;  % number of desired posterior samples
postsamples = zeros(numpostsamples,sum(parmask));

tic
for ii=1:numpostsamples
    ii
    while reject
       totalattempts = totalattempts+1;
       theta = astro_prior_sample;
       bigtheta(1) = log(theta(1));  % the model simulator wants logom not om
       bigtheta(3) = theta(2);
       
       s2 = astro_simsummaries(bigtheta,nobs,nbin);
       simdata_resampled2 = s2(resample_indeces2);
       simdata_resampled2 = sort(simdata_resampled2);
       % notice, in this problem data=summaries
       simsumm2 = simdata_resampled2';
       xc = bsxfun(@minus,simsumm2,s_obs');
       distance = sqrt(sum(xc .* xc, 2));
       index_inclusion1 = distance < 0.45;
       n1test = sum(index_inclusion1); % number of test summaries falling into the ellipsis above
       distance1 = distance(index_inclusion1);
       index_inclusion2 = (distance < 0.55) & ~(distance < 0.45);  % was 2.4 and 2.1
       n2test = sum(index_inclusion2); % number of test summaries falling into the ellipsis above but not in the innermost one 
       distance2 = distance(index_inclusion2);
       index_inclusion3 =  ~(distance < 0.55);
       n3test = sum(index_inclusion3);
       distance3 = distance(index_inclusion3);
       
       if n1test==0 || n2test==0 || n3test==0 % skip the rest of the loop because of neglected strata
           continue
       end

       s1 = astro_simsummaries(bigtheta,nobs,nbin);
       simdata_resampled1 = s1(resample_indeces1);
       simdata_resampled1 = sort(simdata_resampled1);
       % notice, in this problem data=summaries
       simsumm1 = simdata_resampled1';
       xc = bsxfun(@minus,simsumm1,s_obs');
       distance = sqrt(sum(xc .* xc, 2));
       index_inclusion1 = distance < 0.45;
       n1 = sum(index_inclusion1);
       omega1 = n1/numresample1; 
       index_inclusion2 = (distance < 0.55) & ~(distance < 0.45);
       n2 = sum(index_inclusion2);
       omega2 = n2/numresample1;
       omega3 = 1-(omega1+omega2);

       logL1 = log(omega1/n1test) -nsummary*log(ABCthreshold) + logsumexp(-distance1.^2/(2*ABCthreshold^2));
       logL2 = log(omega2/n2test) -nsummary*log(ABCthreshold) + logsumexp(-distance2.^2/(2*ABCthreshold^2));
       logL3 = log(omega3/n3test) -nsummary*log(ABCthreshold) + logsumexp(-distance3.^2/(2*ABCthreshold^2));
       abclogkernel =  logsumexp([logL1,logL2,logL3]);

       if log(rand) <  abclogkernel -log(c_norm_const)
           postsamples(ii,:) = theta;
           break
       end
    end
end
eval_time = toc
histogram(postsamples(:,1))
histogram(postsamples(:,2))
filename = sprintf('postsamples_strat_M=%g',M);
save(filename,'postsamples','-ascii')
totalattempts
filename = sprintf('strat_totalattempts');
save(filename,'totalattempts','-ascii')
filename = sprintf('strat_evaltime');
save(filename,'eval_time','-ascii')