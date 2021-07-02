function out = astro_prior(theta)

% Returns the product of independent priors for parameters of a g-and-k distribution
% Input:  - theta, the vector of parameters to be estimated
% Output: - out, the product of the prior distributions set for each parameter.
%                possibily multiplied with a jacobian for transformations
%                from log-parameter to parameter


  logom = theta(1);
  w0 = theta(2);
 % logh0 = theta(3);

  om = exp(logom);
 % h0 = exp(logh0);

  om_prior      = betapdf(om,3,3);
  w0_prior      = normpdf(w0,-0.5,0.5);
 % h0_prior      = normpdf(h0,0.7,0.02);

  jacobian = om;  % transformation jacobian, since logom --> om

  out = om_prior*w0_prior * jacobian;
end
