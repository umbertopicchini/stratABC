function out = gk_prior(theta)

% Returns the product of independent priors for parameters of a g-and-k distribution
% Input:  - theta, the vector of parameters to be estimated
% Output: - out, the product of the prior distributions set for each
% parameter.

% Umberto Picchini 2016
% www.maths.lth.se/matstat/staff/umberto/

logA = theta(1);
logB = theta(2);
logg = theta(3);
logk = theta(4);

A = exp(logA);
B = exp(logB);
g = exp(logg);
k = exp(logk);

A_prior      = unifpdf(A, -30,30); 
B_prior      = unifpdf(B, 0,30); 
g_prior      = unifpdf(g, 0,30); 
k_prior      = unifpdf(k, 0,30);  

jacobian = A*B*g*k;

out = A_prior*B_prior*g_prior*k_prior * jacobian;




