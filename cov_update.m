function  covupd = cov_update(Xnew,sum_old,m,cov_old)

% On-line covariance matrix update.
% Given a matrix of new data Xnew, the covariance matrix of previous and
% new data is returned. Xnew is an n x d matrix. By-rows sums of the
% previous (old) m x d data are provided via "sum_old". The d x d covariance matrix
% of the old data is provided via cov_old. The function returns the
% covariance matrix of the (m+n) x d matrix stacking old and new data.
%
% Example: xold = randn(6,4);  
%          m    = size(xold,1);
%          xnew = randn(5,4);
%          cov_update(xnew,sum(xold),m,cov(xold))
%          % compare with
%          cov([xold;xnew])

% Reference is eq. (5.3) in Chan, Golub and LeVeque (1979) "Updating formulae and a
% pairwise algorithm for computing sample variances", Tech. Report,
% STAN-CS-79-773, Stanford University.
% ftp://reports.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

% This file is part of the "abc-sde" program. Copyright (C) 2013 Umberto Picchini
% https://sourceforge.net/projects/abc-sde/
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

n = size(Xnew,1);

if n>1
   numcov = (m-1)*cov_old + (n-1)*cov(Xnew) + m/(n*(m+n))*(n/m*sum_old-sum(Xnew))'*(n/m*sum_old-sum(Xnew));
else
   numcov = (m-1)*cov_old + m/(n*(m+n))*(n/m*sum_old-Xnew)'*(n/m*sum_old-Xnew);
end

covupd = numcov/(n+m-1);

