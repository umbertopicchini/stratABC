# stratABC
ABC using stratified Monte Carlo and bootstrapping: supporting code for the paper by 
U. Picchini and R.G. Everitt "Stratified sampling and bootstrapping for approximate Bayesian computation", arXiv:1905.07976

The following folders are included:
- "gauss", illustrating the first case study in arXiv:1905.07976, MATLAB code.
- "gk", illustrating the a case study in Supplementary Material of arXiv:1905.07976, MATLAB and R code.
- "supernova", illustrating a case study in arXiv:1905.07976, MATLAB code.
- "lotka-volterra", illustrating a case study in arXiv:1905.07976, MATLAB code.

Here we describe the content of each folder:
- "gauss"
    - demo_gauss.m: runs pmABC-MCMC
    - demo_gauss_resampling.m: runs rABC-MCMC
    - demo_gauss_stratified_3strata.m: runs rsABC-MCMC
    - demo_gauss_stratified_3strata_averagedlikelihoods.m: runs xrsABC-MCMC
    - demo_gauss_two_independent_sample: runs pmABC-MCMC with M=2
    - subfolder "loglikelihood estimation": produces results for section 6.1.1
    - subfolder "appendix_code": produces results for the Supplementary Material section "Efficiency of the averaged likelihood approach" 
- "gk"
    - subfolder "stratifiedABC" performs a few iterations with rABC-MCMC and then a few more using rsABC-MCMC follow.
    - subfolder "exchanged-likelihoods" performs a few iterations with rABC-MCMC and then a few more using xrsABC-MCMC follow.
    - subfolder "ABCmultiplesamples" performs pmABC-MCMC.
- "lotka-volterra" 
    - subfolder "ABC-SMC" runs sequential Monte Carlo ABC (no resamplig, no stratification) using the algorithm described in Supplementary Material
    - subfolder "pseudomarginalABC_threshold=0.6" runs pmABC-MCMC
    - subfolder "rsABC-MCMC_3strata" runs rsABC-MCMC using three strata.
    - subfolder "several-bootstrap-comparisons" contains results as given in Supplementary Material, comparing the performance of several bootstrap strategies.
    - subfolder "computationally-intensive-model": contains runs of pmABCMCMC and rsABCMCMC for the expensive case study considered in Supplementary Material. 

