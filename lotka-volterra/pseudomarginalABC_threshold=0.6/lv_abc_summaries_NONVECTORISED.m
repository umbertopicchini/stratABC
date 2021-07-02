function summaries = lv_abc_summaries(xhat)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

   % xhat.
   x1 = xhat(:,1:2:end);
   x2 = xhat(:,2:2:end);
   
   mean_x1 = mean(x1);
   mean_x2 = mean(x2);
   var_x1 = var(x1);
   var_x2 = var(x2);
   
   %:::::::: autocorrelations for x1 :::::::::
   % notice, unlike mean and var, unfortunately acf below is not vectorised
   acf_x1_all = zeros(2,size(x1,2));
   for ii=1:size(x1,2)
        try
            acfx1 = acf(x1(:,ii),2,1);
            acf_x1_1 = acfx1(1); % lag 1
            acf_x1_2 = acfx1(2); % lag 2
            acf_x1_all(1,ii) = acf_x1_1;
            acf_x1_all(2,ii) = acf_x1_2;
        catch
              acf_x1_all(1,ii) = NaN;
              acf_x1_all(2,ii) = NaN;
        end
   end
   
   %:::::::: autocorrelations for x2 :::::::::
   % notice, unlike mean and var, unfortunately acf below is not vectorised
   acf_x2_all = zeros(2,size(x2,2));
   for ii=1:size(x2,2)
        try
            acfx2 = acf(x2(:,ii),2,1);
            acf_x2_1 = acfx2(1); % lag 1
            acf_x2_2 = acfx2(2); % lag 2
            acf_x2_all(1,ii) = acf_x2_1;
            acf_x2_all(2,ii) = acf_x2_2;
        catch
              acf_x2_all(1,ii) = NaN;
              acf_x2_all(2,ii) = NaN;
        end
   end
   
  % cross correlation coefficient between x1 and x2
  corrx1x2 = zeros(1,size(x1,2));
  for ii=1:size(x1,2)
        corrx1x2(ii) = corr(x1(:,ii),x2(:,ii));
  end

    
   summaries = [ mean_x1;...
                 mean_x2;...
                 log(var_x1+1);...
                 log(var_x2+1);...
                 acf_x1_all(1,:);...
                 acf_x1_all(2,:);...
                 acf_x2_all(1,:);...
                 acf_x2_all(2,:);...
                 corrx1x2
                ];



end

