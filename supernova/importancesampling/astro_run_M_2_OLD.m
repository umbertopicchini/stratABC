
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
postsamples = zeros(numpostsamples,2);
ABCthreshold = 0.15;
M = 2;

rng(1234)

bigtheta =  parbase;
tStart = tic;
logweights = zeros(1,numpostsamples);
lognormpdf = @(x,m,s)(-0.5*log(2*pi)-log(s)-(x-m)^2 / (2*s^2));
for ii=1:numpostsamples
    ii
       mean_om_is = 0.327;  % mean of the importance sampler for om
       std_om_is = 2*sqrt(0.0127); % std of the importance sampler for om
       om_is = mytruncgaussrandraw([0,1,mean_om_is,std_om_is],rand);
       theta(1) = log(om_is); 
       mean_w0_is = -1.002; % mean of the importance sampler for w0
       std_w0_is = 2*sqrt(0.0776); % std of the importance sampler for w0
       w0_is = normrnd(mean_w0_is ,std_w0_is);
       theta(2) = w0_is;
       bigtheta(1) = theta(1);  % the model simulator wants logom not om
       bigtheta(3)= theta(2);
       s = astro_simsummaries(bigtheta,nobs,nbin);
       xc = bsxfun(@minus,s',s_obs');
       distance = sqrt(sum(xc .* xc, 2));  
       abclogkernel_1 = -distance.^2/(2*ABCthreshold^2);
       s = astro_simsummaries(bigtheta,nobs,nbin);
       xc = bsxfun(@minus,s',s_obs');
       distance = sqrt(sum(xc .* xc, 2));  
       abclogkernel_2 = -distance.^2/(2*ABCthreshold^2);
       logweights(ii) = logsumexp([abclogkernel_1,abclogkernel_2]) -log(2) + log(astro_prior(theta)) -lognormpdf(om_is,mean_om_is,std_om_is)-lognormpdf(w0_is,mean_w0_is,std_w0_is);
       postsamples(ii,:) = [exp(theta(1)),theta(2)];
end
eval_time = toc(tStart);
weights = exp(logweights);
save('eval_time.dat','eval_time','-ascii') 
filename = sprintf('weights_and_samples_threshold=%g_M=%g',ABCthreshold,M);
weights_and_samples = [weights',postsamples];
save(filename,'weights_and_samples','-ascii')
normweights = weights / sum(weights); % normalised weights, to compute the ESS
ESS = 1/(sum(normweights.^2));
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
% let's do w0 now
[w0_sorted,indeces] = sort(postsamples(:,2));
id_lower = find(cumsum(normweights(indeces))<0.025);
id_lower = id_lower(end)+1;
w0_lower_quantile = w0_sorted(id_lower);
id_upper = find(cumsum(normweights(indeces))<0.975);
id_upper = id_upper(end)+1;
w0_upper_quantile = w0_sorted(id_upper);
cred_interval_w0 = [w0_lower_quantile,w0_upper_quantile]