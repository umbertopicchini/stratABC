% In this script we compare different strategies for bootstrapping dependent data. 
% we start with the block bootstrap (BB) with 4 non-overlapping blocks of size 8
% each).
% Then we consioder BB with overlapping blocks.
% Then the circular bootstrap (CB)

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



%:::: load data ::::::::::::::::::::::::::::::::::::::::::::::::::::
load('lv_data.dat')
yobs=lv_data;
summobs = lv_abc_summaries(yobs);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

% load summaries simulated from a prior predictive simulation
load('summaries_pilot.dat')
% remove NaNs
idnan = any(isnan(summaries_pilot),2); % find rows with NaNs
summaries_pilot_fixed = summaries_pilot(~idnan,:);  % remove those
% remove nasty outliers before computing a measure of variation for the
% summaries
summaries_pilot_fixed = rmoutliers(summaries_pilot_fixed,'percentiles',[1.25,98.75]);

% obtain weighting matrix for ABC summaries
summ_weights = diag(mad(summaries_pilot_fixed,0).^2);


big_loop_repetition = 30;
numattempts = 1000; % number of simulated datasets within each repetition
all_percentiles_BB_fixedsizefixedindeces = zeros(big_loop_repetition,5);
all_percentiles_BB_fixedsizerandomindeces = zeros(big_loop_repetition,5);
all_percentiles_OBB_fixedsize = zeros(big_loop_repetition,5);
all_percentiles_statuni_randomsize = zeros(big_loop_repetition,5);

for bigloop = 1: big_loop_repetition
    bigloop
 % HERE WE CONSIDER THE BLOCK BOOTSTRAP WITH 4 NON-OVERLAPPING BLOCKS OF SIZE
% 8 EACH AND CONSTANT SAMPLED INDECES THROUGHOUT.    

prior_pred_numresample = 256;  % it should not be larger than 256 as it would be pointless. There are only 256 different ways to choose 4 blocks with repetitions
nobs = length(sampletime);
resample_indeces = zeros(nobs,prior_pred_numresample);
block1 = [1:nobs/4]';
block2 = [nobs/4+1:nobs/2]';
block3 = [nobs/2+1:nobs*3/4]';
block4 = [nobs*3/4+1:nobs]';
% FIX SAMPLING INDECES ONECE AND FOR ALL
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
simdata_resampled = zeros(nobs,2*prior_pred_numresample);
for ii=1:numattempts
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
          [t,x] = directMethod(stoichiometry', prop_pilot, [sampletime(1),sampletime(end)], xinit', const_rates_pilot);
       catch
          x = NaN*ones(length(sampletime),2);
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
save('alldistances_BB_fixedsize_fixedindeces.dat','alldistances','-ascii')
% let's loogk at all distances based on resampled data
%histogram(alldistances)
%title('BB fixed size 8 fixed indeces')
all_percentiles_BB_fixedsizefixedindeces(bigloop,:) = prctile(alldistances,[0.1:.1:0.5]);
save('all_percentiles_BB_fixedsizefixedindeces.dat','all_percentiles_BB_fixedsizefixedindeces','-ascii')

%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%:::::::: BLOCKED BOOTSTRAP WITH FIXED SIZE 8 AND NON-CONSTANT INDECES

prior_pred_numresample = 256;  % it should not be larger than 256 as it would be pointless. There are only 256 different ways to choose 4 blocks with repetitions
alldistances = [];
simdata_resampled = zeros(nobs,2*prior_pred_numresample);
for ii=1:numattempts
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
          x = NaN*ones(length(sampletime),2);
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
        % RESAMPLE INDECES AT EACH ATTEMPT
        indeces = randsample(4,4,'true');
        count = 0;
        for kk=1:length(indeces)
           if indeces(kk) == 1
            resample_indeces(count+1:count+8,jj) =  block1;
           elseif indeces(kk) == 2
            resample_indeces(count+1:count+8,jj) =  block2; 
           elseif indeces(kk) == 3
            resample_indeces(count+1:count+8,jj) =  block3;
           else
            resample_indeces(count+1:count+8,jj) =  block4; 
           end
           count = count + 8;
        end
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
save('alldistances_BB_fixedsize_randomindeces.dat','alldistances','-ascii')
% let's loogk at all distances based on resampled data
%histogram(alldistances)
%title('BB fixed size 8 and random indeces')
all_percentiles_BB_fixedsizerandomindeces(bigloop,:) = prctile(alldistances,[0.1:.1:0.5]);
save('all_percentiles_BB_fixedsizerandomindeces.dat','all_percentiles_BB_fixedsizerandomindeces','-ascii')

%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%:::::::: HERE WE CONSIDER OVERLAPPING BLOCKS BOOTSTRAP WITH SIZE 8

prior_pred_numresample = 256;  % it should not be larger than 256 as it would be pointless. There are only 256 different ways to choose 4 blocks with repetitions
nobs = length(sampletime);
alldistances = [];
simdata_resampled = zeros(nobs,2*prior_pred_numresample);
for ii=1:numattempts
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
          x = NaN*ones(length(sampletime),2);
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
       simdata_resampled(:,jj+count:jj+1+count) = overlappingBB(xhat,8);
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

% let's look at all distances based on resampled data
save('alldistances_OBB_fixedsize.dat','alldistances','-ascii')
% let's loogk at all distances based on resampled data
%histogram(alldistances)
%title('Overlapping BB fixed size 8')
all_percentiles_OBB_fixedsize(bigloop,:) = prctile(alldistances,[0.1:.1:0.5]);
save('all_percentiles_OBB_fixedsize.dat','all_percentiles_OBB_fixedsize','-ascii')

%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%::: stationary uniform bootstrap with random sizes between 4 and 8

prior_pred_numresample = 256;  % it should not be larger than 256 as it would be pointless. There are only 256 different ways to choose 4 blocks with repetitions
nobs = length(sampletime);
alldistances = [];
simdata_resampled = zeros(nobs-2,2*prior_pred_numresample);
% make observations more stationary by taking order-two differences 
yobsdiff = diff(yobs,2);
summobsdiff = lv_abc_summaries(yobsdiff);

for ii=1:numattempts
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
          x = NaN*ones(length(sampletime),2);
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
    xhatdiff = diff(xhat,2); % order-two differences for enhanced stationarity
    for jj=1:prior_pred_numresample 
       simdata_resampled(:,jj+count:jj+1+count) = stationaryBB(xhatdiff,2,[4;8]);
       count = count +1;
    end
    sim_summaries = feval([problem, '_abc_summaries'],simdata_resampled);

   xc = bsxfun(@minus,sim_summaries',summobsdiff');
   distance = sqrt(sum((xc / summ_weights) .* xc, 2));
   if isnan(distance)
       distance = [];
       alldistances = [alldistances;distance];
   else
       alldistances = [alldistances;distance];
   end
end

% let's look at all distances based on resampled data
save('alldistances_statuni_randomsize.dat','alldistances','-ascii')
% let's loogk at all distances based on resampled data
%histogram(alldistances)
%title('Stationary uniform bootstrap with random size U(4,8)')
all_percentiles_statuni_randomsize(bigloop,:) = prctile(alldistances,[0.1:.1:0.5]);
save('all_percentiles_statuni_randomsize.dat','all_percentiles_statuni_randomsize','-ascii')
end