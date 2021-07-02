
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
parbase =       [     -0.11            0        -0.5         0   logh0_true];

% simulate "observed" summaries
s_obs = astro_simsummaries(bigtheta_true,nobs,nbin);
nsummary = length(s_obs);
%:::: end of data generation :::::::::::::::::::::::::::::::::::::::::::


problem = 'astro';
numpostsamples=1000;  % number of desired posterior samples
postsamples = zeros(numpostsamples,2);
ABCthreshold = 0.15;
M = 1;

rng(1234)

bigtheta =  parbase;
reject = 1;
totalattempts = 0;
tStart = tic;
for ii=1:numpostsamples
    ii
    while reject
       totalattempts = totalattempts+1;
       theta = astro_prior_sample;
       bigtheta(1) = log(theta(1));  % the model simulator wants logom not om
       bigtheta(3)= theta(2);
       s = astro_simsummaries(bigtheta,nobs,nbin);
       xc = bsxfun(@minus,s',s_obs');
       distance = sqrt(sum(xc .* xc, 2));  
       abclogkernel = -distance.^2/(2*ABCthreshold^2);
       if log(rand) < abclogkernel
           postsamples(ii,:) = theta;
           break
       end
    end
end
eval_time = toc(tStart); 
save('eval_time.dat','eval_time','-ascii') 
filename = sprintf('postsamples_threshold=%g_M=%g',ABCthreshold,M);
save(filename,'postsamples','-ascii')
filename = sprintf('totalattempts_threshold=%g_M=%g',ABCthreshold,M);
save(filename,'totalattempts','-ascii')
histogram(postsamples(:,1))
histogram(postsamples(:,2))