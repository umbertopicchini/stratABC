function summaries = lv_abc_summaries(xhat)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

   n = size(xhat,1);

   % xhat.
   x1 = xhat(:,1:2:end);
   x2 = xhat(:,2:2:end);
   
   mean_x1 = mean(x1);
   mean_x2 = mean(x2);
   var_x1 = var(x1);
   var_x2 = var(x2);
   
   %:::::::: lag 1 autocorrelation for x1 :::::::::
   diff1_x1 = (x1(1:n-1,:)-repmat(mean_x1,n-1,1));
   diff2_x1 = (x1(2:n,:)-repmat(mean_x1,n-1,1)); 
   c1_x1 = sum(diff1_x1.*diff2_x1,1);
   autocorr_lag1_x1 = c1_x1 ./ ((n-1)*var_x1);  
   
   %:::::::: lag 1 autocorrelation for x2 :::::::::
   diff1_x2 = (x2(1:n-1,:)-repmat(mean_x2,n-1,1));
   diff2_x2 = (x2(2:n,:)-repmat(mean_x2,n-1,1)); 
   c1_x2 = sum(diff1_x2.*diff2_x2,1);
   autocorr_lag1_x2 = c1_x2 ./ ((n-1)*var_x2);  

   %:::::::: lag 2 autocorrelation for x1 :::::::::
   diff1_x1 = (x1(1:n-2,:)-repmat(mean_x1,n-2,1));
   diff2_x1 = (x1(3:n,:)-repmat(mean_x1,n-2,1)); 
   c2_x1 = sum(diff1_x1.*diff2_x1,1);
   autocorr_lag2_x1 = c2_x1 ./ ((n-1)*var_x1); 
   
   %:::::::: lag 2 autocorrelation for x2 :::::::::
   diff1_x2 = (x2(1:n-2,:)-repmat(mean_x2,n-2,1));
   diff2_x2 = (x2(3:n,:)-repmat(mean_x2,n-2,1)); 
   c2_x2 = sum(diff1_x2.*diff2_x2,1);
   autocorr_lag2_x2 = c2_x2 ./ ((n-1)*var_x2);   
  
  % correlation coefficient between x1 and x2
  ss_x1x2 = sum((x1-repmat(mean_x1,n,1)).*(x2-repmat(mean_x2,n,1)),1);
  ss_x1 = sum((x1-repmat(mean_x1,n,1)).^2,1);
  ss_x2 = sum((x2-repmat(mean_x2,n,1)).^2,1);
  corr_x1x2 = ss_x1x2 ./ sqrt(ss_x1.*ss_x2);

    
   summaries = [mean_x1;...
                 mean_x2;...
                 log(var_x1+1);...
                 log(var_x2+1);...
                 autocorr_lag1_x1;...
                 autocorr_lag2_x1;...
                 autocorr_lag1_x2;...
                 autocorr_lag2_x2;...
                 corr_x1x2
                ];



end

