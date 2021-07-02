function summaries = lv_abc_summaries(xhat)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

   % xhat.
   x1 = xhat(:,1);
   x2 = xhat(:,2);
   
   mean_x1 = mean(x1);
   mean_x2 = mean(x2);
   var_x1 = var(x1);
   var_x2 = var(x2);
   
   %:::::::: autocorrelations for x1 :::::::::
   try
      acfx1 = acf(x1,2,1);
      acf_x1_1 = acfx1(1); % lag 1
      acf_x1_2 = acfx1(2); % lag 2
   catch
      acf_x1_1 = NaN;
      acf_x1_2 = NaN;
   end
   
   %:::::::: autocorrelations for x2 :::::::::
   try
      acfx2 = acf(x2,2,1);
      acf_x2_1 = acfx2(1); % lag 1
      acf_x2_2 = acfx2(2); % lag 2  
   catch
      acf_x2_1 = NaN;
      acf_x2_2 = NaN;  
   end
   
  % cross correlation coefficient between x1 and x2
  corrx1x2 = corr(x1,x2);

    
   summaries = [ mean_x1;...
                 mean_x2;...
                 log(var_x1+1);...
                 log(var_x2+1);...
                 acf_x1_1;...
                 acf_x1_2;...
                 acf_x2_1;...
                 acf_x2_2;...
                 corrx1x2
                ];



end

