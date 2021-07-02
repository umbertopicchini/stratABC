load('loglike_vec')  % load loglikelihoods obtained with 1000 repetitions of rsABC-MCMC
loglike_vec_rs = loglike_vec;
load('loglike_vec_xrs') % load loglikelihoods obtained with 1000 repetitions of xrsABC-MCMC
loglike_vec_xrs = loglike_vec;

figure
mu_vec = linspace(-0.1,0.1,50);   % 50 parameter points to try in the interval [-0.1,0.1]. Notice the interval includes the true value mu=0
plot(mu_vec,var(exp(loglike_vec_rs)),'*k--',mu_vec,var(exp(loglike_vec_xrs)),'ok--')
xlabel('\theta')
%axis([-0.1 0.1 0 22])

figure
plot(mu_vec,100*(1-var(exp(loglike_vec_xrs))./var(exp(loglike_vec_rs))),'ok--')
xlabel('\theta')
ylabel('100*(1-varLik/varAvgLik)')
%axis([-0.1 0.1 0 44])