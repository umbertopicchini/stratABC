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
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

%::: ABC PILOT to collect simulated summaries and compute a matrix that weights them

% THE PART BELOW CAN BE SKIPPED BY JUST LOADING THE ATTACHED
% SUMMARIES_PILOT MATRIX, THAT IS YOU CAN JUN RUN
% load('summaries_pilot.dat')

size_pilot = 5000;
summaries_pilot = zeros(size_pilot,9);
abc_distances = zeros(size_pilot,1);
summobs = lv_abc_summaries(yobs);
for ii=1:size_pilot
    ii
    try
      logc1_pilot   = unifrnd(-5,2); 
      logc2_pilot   = unifrnd(-5,2); 
      logc3_pilot   = unifrnd(-5,2);
      const_rates_pilot = [exp(logc1_pilot) exp(logc2_pilot) exp(logc3_pilot)]; % these are the (c1,c2,c3) constant rates
      prop_pilot = @(x,const_rates_pilot)([const_rates_pilot(1).*x(:,1),...
                 const_rates_pilot(2).*x(:,1).*x(:,2),...
                 const_rates_pilot(3).*x(:,2)]);
 %  xhat = lv_model(bigtheta_pilot,sampletime,stoichiometry,1);
      [t,x] = directMethod(stoichiometry', prop_pilot, [sampletime(1),sampletime(end)], xinit', const_rates_pilot);
   catch
       try
          logc1_pilot   = unifrnd(-5,2); 
          logc2_pilot   = unifrnd(-5,2); 
          logc3_pilot   = unifrnd(-5,2);
          const_rates_pilot = [exp(logc1_pilot) exp(logc2_pilot) exp(logc3_pilot)]; % these are the (c1,c2,c3) constant rates
          prop_pilot = @(x,const_rates_pilot)([const_rates_pilot(1).*x(:,1),...
          const_rates_pilot(2).*x(:,1).*x(:,2),...
          const_rates_pilot(3).*x(:,2)]);
 %  xhat = lv_model(bigtheta_pilot,sampletime,stoichiometry,1);
          [t,x] = directMethod(stoichiometry', prop_pilot, [sampletime(1),sampletime(end)], xinit', const_rates_pilot);
       catch
          logc1_pilot   = unifrnd(-5,2); 
          logc2_pilot   = unifrnd(-5,2); 
          logc3_pilot   = unifrnd(-5,2);
          const_rates_pilot = [exp(logc1_pilot) exp(logc2_pilot) exp(logc3_pilot)]; % these are the (c1,c2,c3) constant rates
          prop_pilot = @(x,const_rates_pilot)([const_rates_pilot(1).*x(:,1),...
          const_rates_pilot(2).*x(:,1).*x(:,2),...
          const_rates_pilot(3).*x(:,2)]);
          [t,x] = directMethod(stoichiometry', prop_pilot, [sampletime(1),sampletime(end)], xinit', const_rates_pilot);
       end
   end
   % subselect the output
   try
       xhat=interp1(t,x,sampletime);
   catch
       xhat = NaN*ones(length(sampletime),2);
   end
   sim_summaries = lv_abc_summaries(xhat);
   summaries_pilot(ii,:) = sim_summaries;
%    xc = bsxfun(@minus,sim_summaries',summobs');
%    distance = sqrt(sum(xc .* xc, 2));
%    abc_distances(ii) = distance;
end

save('summaries_pilot.dat','summaries_pilot','-ascii')
%:::::::::: END OF PILOT :::::::::::::::::::::::::::::::::::::::::::::
return

% remove NaNs
idnan = any(isnan(summaries_pilot),2); % find rows with NaNs
summaries_pilot_fixed = summaries_pilot(~idnan,:);  % remove those
% remove nasty outliers before computing a measure of variation for the
% summaries
summaries_pilot_fixed = rmoutliers(summaries_pilot_fixed,'percentiles',[1.25,98.75]);

% obtain weighting matrix for ABC summaries
summ_weights = diag(mad(summaries_pilot_fixed,0).^2);

return

% simulate distances from the prior predictive.
% notice, to have a realistic picture to be used with resampling methods
% we need to randomize the output, or otherwise distances will look
% smaller than they will be when using resampling 


prior_pred_numresample = 256;  % it should not be larger than 256 as it would be pointless. There are only 256 different ways to choose 4 blocks with repetitions
nobs = length(sampletime);
resample_indeces = zeros(nobs,prior_pred_numresample);
block1 = [1:nobs/4]';
block2 = [nobs/4+1:nobs/2]';
block3 = [nobs/2+1:nobs*3/4]';
block4 = [nobs*3/4+1:nobs]';
for jj=1:prior_pred_numresample
     indeces = randsample(4,4,'true');
     count = 0;
     for ii=1:length(indeces)
         if indeces(ii) == 1
            resample_indeces(count+1:count+8,jj) =  block1;
         elseif indeces(ii) == 2
            resample_indeces(count+1:count+8,jj) =  block2; 
         elseif indeces(ii) == 3
            resample_indeces(count+1:count+8,jj) =  block3;
         else
            resample_indeces(count+1:count+8,jj) =  block4; 
         end
         count = count + 8;
     end
end
alldistances = [];
numattempts = 1000;
simdata_resampled = zeros(nobs,2*prior_pred_numresample);
for ii=1:numattempts
    ii
    try
      logc1_pilot   = unifrnd(-5,2); 
      logc2_pilot   = unifrnd(-5,2); 
      logc3_pilot   = unifrnd(-5,2);
      const_rates_pilot = [exp(logc1_pilot) exp(logc2_pilot) exp(logc3_pilot)]; % these are the (c1,c2,c3) constant rates
      prop_pilot = @(x,const_rates_pilot)([const_rates_pilot(1).*x(:,1),...
                 const_rates_pilot(2).*x(:,1).*x(:,2),...
                 const_rates_pilot(3).*x(:,2)]);
 %  xhat = lv_model(bigtheta_pilot,sampletime,stoichiometry,1);
      [t,x] = directMethod(stoichiometry', prop_pilot, [sampletime(1),sampletime(end)], xinit', const_rates_pilot);
   catch
       try
          logc1_pilot   = unifrnd(-5,2); 
          logc2_pilot   = unifrnd(-5,2); 
          logc3_pilot   = unifrnd(-5,2);
          const_rates_pilot = [exp(logc1_pilot) exp(logc2_pilot) exp(logc3_pilot)]; % these are the (c1,c2,c3) constant rates
          prop_pilot = @(x,const_rates_pilot)([const_rates_pilot(1).*x(:,1),...
          const_rates_pilot(2).*x(:,1).*x(:,2),...
          const_rates_pilot(3).*x(:,2)]);
 %  xhat = lv_model(bigtheta_pilot,sampletime,stoichiometry,1);
          [t,x] = directMethod(stoichiometry', prop_pilot, [sampletime(1),sampletime(end)], xinit', const_rates_pilot);
       catch
          logc1_pilot   = unifrnd(-5,2); 
          logc2_pilot   = unifrnd(-5,2); 
          logc3_pilot   = unifrnd(-5,2);
          const_rates_pilot = [exp(logc1_pilot) exp(logc2_pilot) exp(logc3_pilot)]; % these are the (c1,c2,c3) constant rates
          prop_pilot = @(x,const_rates_pilot)([const_rates_pilot(1).*x(:,1),...
          const_rates_pilot(2).*x(:,1).*x(:,2),...
          const_rates_pilot(3).*x(:,2)]);
          [t,x] = directMethod(stoichiometry', prop_pilot, [sampletime(1),sampletime(end)], xinit', const_rates_pilot);
       end
   end
   % subselect the output
   try
       xhat=interp1(t,x,sampletime);
   catch
       xhat = NaN*ones(length(sampletime),2);
   end
    % resample data    
    count = 0;
    for jj=1:prior_pred_numresample 
       simdata_resampled(:,jj+count:jj+1+count) = xhat(resample_indeces(:,jj),:);
       count = count +1;
    end
    sim_summaries = feval([problem, '_abc_summaries'],simdata_resampled);

   xc = bsxfun(@minus,sim_summaries',summobs');
   distance = sqrt(sum((xc / summ_weights) .* xc, 2));
   if isnan(distance)
       distance = [];
       alldistances = [alldistances;distance];
   else
       alldistances = [alldistances;distance];
   end
end

% let's loogk at all distances based on resampled data
histogram(alldistances)
prctile(alldistances,[0.1:0.01:0.5])
return

rng(200)

                  % log_x10  log_x20      log_c1   log_c2   log_c3  
bigtheta_start = [  log(50)  log(100)    log(1)  log(0.008)  log(0.6)  ];
parmask        = [        0        0         1         1          1     ];

parbase = bigtheta_start;


% ABC settings
numresample1 = 256; % it should not be larger than 256 as it would be pointless. There are only 256 different ways to choose 4 blocks with repetitions
numresample2 = 256; % it should not be larger than 256 as it would be pointless. There are only 256 different ways to choose 4 blocks with repetitions
ABCthreshold = 0.6*8;
burnin = 500;
length_CoVupdate = 100;
R_mcmc = 20000;
step_rw = [0.02 0.02 0.02];

tic
ABCMCMC = rsabcmcmc(problem,yobs,bigtheta_start,sampletime,stoichiometry,parmask,parbase,numresample1,numresample2,R_mcmc,step_rw,burnin,length_CoVupdate,summ_weights,ABCthreshold);
eval = toc
save('eval_time.dat','eval','-ascii') 


