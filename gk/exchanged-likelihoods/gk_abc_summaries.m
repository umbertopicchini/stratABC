function summariesy  = gk_abc_summaries(y)
% computation of summary statistics
% Input: the "data" y (these are either the actual data or a realization
%                      from the assumed data-generating model).
% Output: a vector of summaries


summariesy = zeros(4,size(y,2));

S_50 = prctile(y,50);
S_25 = prctile(y,25);
S_75 = prctile(y,75);

% the following summaries are from Drovandi and Pettitt 2011, Likelihood-free Bayesian estimation of multivariate quantile distributions

   summariesy(1,:) = S_50; 
   summariesy(2,:) = S_75-S_25; 
   summariesy(3,:) = (S_75+S_25-2*summariesy(1,:))./summariesy(2,:);
   summariesy(4,:) = (prctile(y,87.5)-prctile(y,62.5)+prctile(y,37.5)-prctile(y,12.5))./summariesy(2,:); 


end
    


