function Zb = seasBB(Z,s,b)
% PURPOSE: Seasonal Block Bootstrap for a vector time series
% ------------------------------------------------------------
% SYNTAX: Zb = seasBB(Z,s,b);
% ------------------------------------------------------------
% OUTPUT: Zb : ([N/b]*b*s)xkz resampled time series (N=[n/s])
% ------------------------------------------------------------
% INPUT:  Z :  nxkz --> vector time series to be resampled
%         s :  1x1  --> seasonal frequency (s>1)
%         b :  1x1  --> block size (b>=1)
% ------------------------------------------------------------
% LIBRARY: loopSBB [internal]
% ------------------------------------------------------------
% SEE ALSO: stationaryBB, overlappingBB
% ------------------------------------------------------------
% REFERENCE: Politis, D. (2001) "Resampling time series with 
% seasonal components", "Frontiers in Data Mining and Bioinformatics".
% ------------------------------------------------------------

% written by:
%  Enrique M. Quilis
%  Macroeconomic Research Department
%  Fiscal Authority for Fiscal Responsibility (AIReF)
%  <enrique.quilis@airef.es>

% Version 2.1 [October 2015]

% Checks
if (s <= 1)
    error ('*** SEASONAL FREQUENCY s SHOULD BE GREATER THAN 1 ***');
end

% Dimension of time series to be bootstrapped
[n,kz] = size(Z);

% Number of years
N = fix(n/s);

% Number of blocks (in years)
k = fix(N/b);

% ------------------------------------------------------------
%  ALLOCATION
% ------------------------------------------------------------
I = zeros(1,k);

% ------------------------------------------------------------
% INDEX SELECTION
% ------------------------------------------------------------
I = round(1+(N-b-1)*rand(1,k));

% ------------------------------------------------------------
% BOOTSTRAP REPLICATION
% ------------------------------------------------------------
Zb = [];
for j=1:kz
   Zb = [Zb loopSBB(Z(:,j),s,k,b,I)];
end;

% ============================================================
% loopBB ==> UNIVARIATE BOOTSTRAP LOOP
% ============================================================
function xb = loopSBB(x,s,k,b,I);

for m=1:k
   for j=1:b*s	      
      xb((m-1)*b*s+j) = x(I(m)*s+j);             
   end  
end

xb = xb';