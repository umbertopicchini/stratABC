
rng(1234)  % was rng(1234)

% ground-truth parameters
om = 0.3;
ok = 0;
w0 = -1;
wa = 0;
h0 = 0.7;

logom = log(om); % -0.52
logh0 = log(h0); % -0.357



nobs = 10000;  % data sample size
nbin = 20;
bigtheta_true = [logom, ok, w0, wa, logh0];  % ground-truth parameters

% simulate "observed" summaries
s_obs = astro_simsummaries(bigtheta_true,nobs,nbin);

%:::: end of data generation :::::::::::::::::::::::::::::::::::::::::::

% simulate distances from the prior predictive.
% notice, to have a realistic picture to be used with resampling methods
% we need to randomize the output, or otherwise distances will look
% smaller than they will be when using resampling (this is because the
% simulator produces "centres" of the histogram bins, which are of course
% sorted in increasing order).

numattempts = 2000;
alldistances = zeros(numattempts,1);
prior_pred_numresample = 2000;  % just for the purpose of producing some distances on resampled data
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
     bigtheta = [theta(1),ok,theta(2),wa,logh0];
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
return

%         logom      ok      w0   wa   logh0
parbase = [ -0.11     0    -0.5    0   logh0];
parmask = [   1       0       1    0     0  ];

bigtheta_start = bigtheta_true;

problem = 'astro';
R_mcmc = 15000;  % Number of iterations for ABC-MCMC
numresample1 = 3000; % number of data resamples used to compute the strata probabilities omega_j
numresample2 = 3000;  % number of data resamples used to compute the number of samples n_j for each stratum
step_rw = [0.04, 0.04];
burnin = 500;
ABCthreshold =  0.15; % was 0.07   
length_CoVupdate = 100; % how often is the Haario's adaptive metropolis update applied

if burnin < length_CoVupdate
    error('the burnin must be longer than length_CoVupdate. ')
end

rng(789)
tStart = tic;
chains = abcmcmc(problem,s_obs,nobs,nbin,bigtheta_start,parmask,parbase,ABCthreshold,numresample1,numresample2,R_mcmc,burnin,step_rw,length_CoVupdate);
eval_time = toc(tStart); 
save('eval_time.dat','eval_time','-ascii') 
filename = sprintf('chains');
save(filename,'chains','-ascii')


figure
subplot(2,2,1)
plot(exp(chains(:,1))) % exponentiate logom --> om
hline(om)
subplot(2,2,2)
plot(chains(:,2))
hline(w0)
% subplot(2,2,3)
% plot(exp(chains(:,3))) % exponentiate logh0 --> h0
% hline(h0)
