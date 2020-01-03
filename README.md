# stratABC-MCMC
ABC-MCMC using stratified Monte Carlo and bootstrapping: supporting code for the paper by 
U. Picchini and R.G. Everitt "Stratified sampling and bootstrapping for approximate Bayesian computation", arXiv:1905.07976

The following folders are included:
- "gauss", illustrating the first case study in arXiv:1905.07976, MATLAB code.
- "gk", illustrating the second case study in arXiv:1905.07976, MATLAB code.
- "lotka-volterra", illustrating the fourth case study in arXiv:1905.07976, MATLAB code.

Here we describe the content of each folder:
- "gauss"
    - demo_gauss.m: runs pmABC-MCMC
    - demo_gauss_resampling.m: runs rABC-MCMC
    - demo_gauss_stratified_3strata.m: runs rsABC-MCMC
    - demo_gauss_stratified_3strata_averagedlikelihoods.m: runs xrsABC-MCMC
    - demo_gauss_two_independent_sample: runs pmABC-MCMC with M=2
    - subfolder "loglikelihood estimation": produces results for section 6.1.1
    - subfolder "appendix_code": produces results for the appendix section "Efficiency of the averaged likelihood approach" 
- "gk"
    - subfolder "exactMCMC_R" performs exact Bayesian inference (not ABC) using the R package "gk".
    - subfolder "stratifiedABC" performs a few iterations with rABC-MCMC and then a few more using rsABC-MCMC follow.
    - subfolder "exchanged-likelihoods" performs a few iterations with rABC-MCMC and then a few more using xrsABC-MCMC follow.
    - subfolder "ABCmultiplesamples" performs pmABC-MCMC.
- "lotka-volterra" **IS FORTHCOMING**
    

