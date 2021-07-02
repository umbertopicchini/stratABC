function out = lv_prior(theta)

% Returns the product of independent priors 
% Input:  - theta, the vector of parameters to be estimated
% Output: - out, the product of the prior distributions set for each
% parameter.


logc1 = theta(1);
logc2 = theta(2);
logc3 = theta(3);


logc1_prior      = unifpdf(logc1, -5,2); 
logc2_prior      = unifpdf(logc2, -5,2); 
logc3_prior      = unifpdf(logc3, -5,2); 


out = logc1_prior*logc2_prior*logc3_prior ;

