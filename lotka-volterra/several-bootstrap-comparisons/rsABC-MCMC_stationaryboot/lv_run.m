rng(100)  % was 100 for reproducibility

log_x10 = log(50);   % true parameter value used for data simulation
log_x20 = log(100);   % true parameter value used for data simulation
log_c1  = log(1);   % true parameter value used for data simulation
log_c2  = log(0.008);% true parameter value used for data simulation
log_c3  = log(0.6);   % true parameter value used for data simulation

sampletime = [0:2:62];  % the vector of sampling times
problem = 'lv';

bigtheta_true = [log_x10,log_x20,log_c1,log_c2,log_c3];   % store here all parameters needed for SDE simulation

%DEFINING PROPENSITY FUNCTIONS AS A FUNCTION HANDLE IN VECTORIZED MANNER
const_rates = [exp(log_c1) exp(log_c2) exp(log_c3)]; % these are the (c1,c2,c3) constant rates
prop = @(x,const_rates)([const_rates(1).*x(:,1),...
                 const_rates(2).*x(:,1).*x(:,2),...
                 const_rates(3).*x(:,2)]);
stoichiometry = [1 -1 0; 0 1 -1];
xinit = [exp(log_x10);exp(log_x20)];



%:::: generate data ::::::::::::::::::::::::::::::::::::::::::::::::::::
% or you can load it. It is attached as lv_data.dat
[t,x] = directMethod(stoichiometry', prop, [sampletime(1),sampletime(end)], xinit', const_rates);
% subselect the output
yobs=interp1(t,x,sampletime);
save('lv_data.dat','yobs','-ascii')
% make observations more stationary by taking order-two differences, so we
% can use the stationary bootstrap
yobsdiff = diff(yobs,2);
summobsdiff = lv_abc_summaries(yobsdiff);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

load('summaries_pilot.dat')
% remove NaNs
idnan = any(isnan(summaries_pilot),2); % find rows with NaNs
summaries_pilot_fixed = summaries_pilot(~idnan,:);  % remove those
% remove nasty outliers before computing a measure of variation for the
% summaries
summaries_pilot_fixed = rmoutliers(summaries_pilot_fixed,'percentiles',[1.25,98.75]);

% obtain weighting matrix for ABC summaries
summ_weights = diag(mad(summaries_pilot_fixed,0).^2);

rng(200)

                  % log_x10  log_x20      log_c1   log_c2   log_c3  
bigtheta_start = [  log(50)  log(100)    log(1)  log(0.008)  log(0.6)  ];
parmask        = [        0        0         1         1          1     ];

parbase = bigtheta_start;


% ABC settings
numresample1 = 256; % it should not be larger than 256 as it would be pointless. There are only 256 different ways to choose 4 blocks with repetitions
numresample2 = 256; % it should not be larger than 256 as it would be pointless. There are only 256 different ways to choose 4 blocks with repetitions
ABCthreshold = 1;
burnin = 500;
length_CoVupdate = 100;
R_mcmc = 20000;
step_rw = [0.02 0.02 0.02];

tic
ABCMCMC = rsabcmcmc_statboot(problem,yobsdiff,bigtheta_start,sampletime,stoichiometry,parmask,parbase,numresample1,numresample2,R_mcmc,step_rw,burnin,length_CoVupdate,summ_weights,ABCthreshold);
eval = toc
save('eval_time.dat','eval','-ascii') 


