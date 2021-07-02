numsims = 59;
paramplot = 1;  % the index pf the parameter you want boxplots for
theta = [];
g = [];
for ii = 1:numsims
    filename=sprintf('ABCdraws_stage%d.mat',ii);
    logtheta_temp = load(filename);
    logtheta = logtheta_temp.ABCdraws';
    logtheta = logtheta(:,paramplot);
    theta_temp = exp(logtheta);
    theta = [theta;theta_temp];
    g = [g; ii*ones(size(logtheta))];
end
figure
boxplot(theta,g)
figure
boxplot(theta,g,'PlotStyle','compact') % improves display (eg the median) for figures with many boxplots

%:::::::::::::::::::::::::::::::::::::::::::::::::::::::
% PLOTS ON LOG-SCALE
numsims = 51;
paramplot = 3;  % the index pf the parameter you want boxplots for
logtheta = [];
g = [];
for ii = 1:numsims
    filename=sprintf('ABCdraws_strat_stage%d.mat',ii);
    logtheta_temp = load(filename);
    logtheta_temp = logtheta_temp.ABCdraws';
    logtheta_par = logtheta_temp(:,paramplot);
    logtheta = [logtheta;logtheta_par];
    g = [g; ii*ones(size(logtheta_par))];
end
%figure
boxplot(logtheta,g)
xticks([1:2:numsims])
%xticklabels({'1','3','5','7','9','11','13','15',    '17',    '19',  '21',    '23',    '25',    '27',    '29',    '31',    '33',    '35',    '37','39',    '41'  ,  '43'  ,  '45' ,   '47' ,   '49' ,   '51'  ,  '53'  ,  '55' ,   '57'  ,  '59'})
xticklabels({'1','3','5','7','9','11','13','15',    '17',    '19',  '21',    '23',    '25',    '27',    '29',    '31',    '33',    '35',    '37','39',    '41'  ,  '43'  ,  '45' ,   '47' ,   '49' ,   '51' })
%figure
%boxplot(logtheta,g,'PlotStyle','compact') % improves display (eg the median) for figures with many boxplots