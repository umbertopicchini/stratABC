function xhat = lv_model(bigtheta,sampletime,stoichiometry,numsim)


% Parameters
log_x10 = bigtheta(1);
log_x20 = bigtheta(2);
log_c1  = bigtheta(3);
log_c2  = bigtheta(4);
log_c3  = bigtheta(5);


xinit = [exp(log_x10);exp(log_x20)];

const_rates = [exp(log_c1) exp(log_c2) exp(log_c3)]; % these are the (c1,c2,c3) constant rates
prop = @(x,const_rates)([const_rates(1).*x(:,1),...
                 const_rates(2).*x(:,1).*x(:,2),...
                 const_rates(3).*x(:,2)]);


%trajectory = SSAv(sampletime, const_rates,prop,stoichiometry, xinit, numsim );
%x1 = squeeze(trajectory(:,1,:));
% extract all simulations for the second coordinate
%x2 = squeeze(trajectory(:,2,:));
%xhat = [x1,x2];

try
   [t,x] = directMethod(stoichiometry', prop, [sampletime(1),sampletime(end)], xinit', const_rates);
catch
    xhat = NaN*ones(length(sampletime),2);
    return
end

try
  % subselect the output
  xhat=interp1(t,x,sampletime);
catch
  xhat = NaN*ones(length(sampletime),2);  % just because the interp1 function occasionally throws an error "Error using griddedInterpolant. The grid vectors must contain unique points".
end

end


