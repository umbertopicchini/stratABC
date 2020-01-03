function y = gk_modelsimulate(bigtheta,n,numsim)
%simulate a vector of length n of independent realizations from the g-and-k distributon
% Input: - bigtheta: vector of parameters defining a g-and-k distribution
%        - n: the number of independent draws (realizations) from said distribution
% Output - y: a vector of length n 

% Umberto Picchini 2016
% www.maths.lth.se/matstat/staff/umberto/

logA = bigtheta(1);
logB = bigtheta(2);
logg = bigtheta(3);
logk = bigtheta(4);

A=exp(logA);
B=exp(logB);
g=exp(logg);
k=exp(logk);

z=randn(n,numsim);


y = A + B * (1 + 0.8 * (1-exp(-g*z))./(1+exp(-g*z))) .* (1 + z.^2).^k .* z;


end

