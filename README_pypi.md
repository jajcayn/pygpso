# pyGPSO

Visit the project's [github page](https://github.com/jajcayn/pygpso).

`pyGPSO` is a python package for Gaussian-Processes Surrogate Optimisation. GPSO is a Bayesian optimisation method designed to cope with costly, high-dimensional, non-convex problems by switching between exploration of the parameter space (using partition tree) and exploitation of the gathered knowledge (by training the surrogate function using Gaussian Processes regression). The motivation for this method stems from the optimisation of large-scale biophysical models in neuroscience when the modelled data should match the experimental one. This package leverages [`GPFlow`](https://github.com/GPflow/GPflow) for training and predicting the Gaussian Processes surrogate.

This is port of original [Matlab implementation](https://github.com/jhadida/gpso) by the paper's author.

**Reference**: Hadida, J., Sotiropoulos, S. N., Abeysuriya, R. G., Woolrich, M. W., & Jbabdi, S. (2018). Bayesian Optimisation of Large-Scale Biophysical Networks. NeuroImage, 174, 219-236.
