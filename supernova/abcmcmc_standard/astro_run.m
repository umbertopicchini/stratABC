
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
s_obs = astro_simsummaries(bigtheta_true,nobs,nbin,1);


%:::: end of data generation :::::::::::::::::::::::::::::::::::::::::::

% set parameters starting values

% %         logom      ok      w0   wa   h0
% parbase = [ -0.11     0    -0.5    0    0.7];
% parmask = [   1       0       1    0     0 ];

%         logom      ok      w0   wa   logh0
parbase = [ -0.11     0    -0.5    0   logh0];
parmask = [   1       0       1    0     0  ];

bigtheta_start = bigtheta_true;

problem = 'astro';
R_mcmc = 25000;  % Number of iterations for ABC-MCMC
step_rw = [0.03, 0.03];
burnin = 500;
numsim = 2; % the number of model simulations for each parameter proposal
ABCthreshold = 0.07;   
length_CoVupdate = 100; % how often is the Haario's adaptive metropolis update applied

if burnin < length_CoVupdate
    error('the burnin must be longer than length_CoVupdate. ')
end


rng(789)
tStart = tic;
chains = abcmcmc(problem,s_obs,nobs,nbin,bigtheta_start,parmask,parbase,ABCthreshold,numsim,R_mcmc,burnin,step_rw,length_CoVupdate);
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
