function out = mytruncgaussrandraw(distribParams, myuniforms)
% out =  mytruncgaussrandraw([a,b,mu,sigma], sampleSize) 
% Generate sampleSize draws on the interval [a,b] from a truncated Gaussian with
% mean mu and standard deviation sigma, via "myuniforms", a supplied vector of uniform draws in [0,1].

%
% Umberto Picchini's modification of the multi-purpose "randraw" function (version
% 6 March 2013) http://se.mathworks.com/matlabcentral/fileexchange/7309-randraw
% This stripped-down version only sample variates from a truncated Gaussian



% THE TRUNCATED NORMAL DISTRIBUTION
%
                    %   pdf(y) = normpdf((y-mu)/sigma) / (sigma*(normcdf((b-mu)/sigma)-normcdf((a-mu)/sigma))); a<=y<=b; 
                    %   cdf(y) = (normcdf((y-mu)/sigma)-normcdf((a-mu)/sigma)) / (normcdf((b-mu)/sigma)-normcdf((a-mu)/sigma)); a<=y<=b;
                    %      where mu and sigma are the mean and standard deviation of the parent normal 
                    %            distribution and a and b are the lower and upper truncation points. 
                    %            normpdf and normcdf are the PDF and CDF for the standard normal distribution respectvely
                    %            ( run randraw('normal') for help).
                    %                                        
                    %   Mean = mu - sigma*(normpdf((b-mu)/sigma)-normpdf((a-mu)/sigma))/(normcdf((b-mu)/sigma)-normcdf((a-mu)/sigma));
                    %   Variance = sigma^2 * ( 1 - ((b-mu)/sigma*normpdf((b-mu)/sigma)-(a-mu)/sigma*normpdf((a-mu)/sigma))/(normcdf((b-mu)/sigma)-normcdf((a-mu)/sigma)) - ...
                    %                           ((normpdf((b-mu)/sigma)-normpdf((a-mu)/sigma))/(normcdf((b-mu)/sigma)-normcdf((a-mu)/sigma)))^2 );
                    %
                    % PARAMETERS:  
                    %   a  - lower truncation point;
                    %   b  - upper truncation point; (b>=a)
                    %   mu - Mean of the parent normal distribution
                    %   sigma - standard deviation of the parent normal distribution (sigma>0)
                    %   
                    %
                    % SUPPORT:      
                    %   y,   a <= y <= b
                    %
                    % CLASS:
                    %   Continuous distributions
                    %
                    % USAGE:
                    %   randraw('normaltrunc', [a, b, mu, sigma], sampleSize) - generate sampleSize number
                    %         of variates from Truncated Normal distribution on the interval (a, b) with
                    %         parameters 'mu' and  'sigma';
                    %   randraw('normaltrunc') - help for Truncated Normal distribution;
                    %
                    % EXAMPLES:
                    %  1.   y = randraw('normaltrunc', [0, 1, 0, 1], [1 1e5]);
                    %  2.   y = randraw('normaltrunc', [0, 1, 10, 3], 1, 1e5);
                    %  3.   y = randraw('normaltrunc', [-10, 10, 0, 1], 1e5 );
                    %  4.   y = randraw('normaltrunc', [-13.1, 15.2, 20.1, 3.3], [1e5 1] );
                    %  5.   randraw('normaltrunc');                    
                    % END normaltrunc HELP END normaltruncated HELP END gausstrunc HELP
                    
                    % See http://www.econ.umn.edu/~kortum/courses/fall02/lecture4k.pdf
                    %     http://hydrology.ifas.ufl.edu/publications/jawitz_2004_AWR.pdf
                   
                    
                    a = distribParams(1);
                    b = distribParams(2);
                    mu = distribParams(3);
                    sigma = distribParams(4);
                  
                    PHIl = normcdf((a-mu)/sigma);  
                    PHIr = normcdf((b-mu)/sigma);
                    
                    out = mu + sigma*( sqrt(2)*erfinv(2*(PHIl+(PHIr-PHIl)*myuniforms)-1) );
                    


end 

