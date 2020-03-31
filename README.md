[![Build Status](https://travis-ci.com/jajcayn/pygpso.svg?branch=master)](https://travis-ci.com/jajcayn/pygpso) ![](https://img.shields.io/github/v/release/jajcayn/pygpso) [![codecov](https://codecov.io/gh/jajcayn/pygpso/branch/master/graph/badge.svg)](https://codecov.io/gh/jajcayn/pygpso) [![PyPI license](https://img.shields.io/pypi/l/pygpso.svg)](https://pypi.python.org/pypi/pygpso/) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jajcayn/pygpso.git/master?filepath=examples) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# pyGPSO
*Optimise anything (but mainly large-scale biophysical models) using Gaussian Processes surrogate*

`pyGPSO` is a python package for Gaussian-Processes Surrogate Optimisation. GPSO is a Bayesian optimisation method designed to cope with costly, high-dimensional, non-convex problems by switching between exploration of the parameter space (using partition tree) and exploitation of the gathered knowledge (by training the surrogate function using Gaussian Processes regression). The motivation for this method stems from the optimisation of large-scale biophysical models in neuroscience when the modelled data should match the experimental one. This package leverages [`GPFlow`](https://github.com/GPflow/GPflow) for training and predicting the Gaussian Processes surrogate.

This is port of original [Matlab implementation](https://github.com/jhadida/gpso) by the paper's author.

**Reference**: Hadida, J., Sotiropoulos, S. N., Abeysuriya, R. G., Woolrich, M. W., & Jbabdi, S. (2018). Bayesian Optimisation of Large-Scale Biophysical Networks. NeuroImage, 174, 219-236.

Comparison of the GPR surrogate and the true objective function after optimisation.
<p align="center">
  <img src="resources/example_GPRsurrogate.svg" width="720">
</p>

Example of ternary partition tree after optimisation.
<p align="center">
  <img src="resources/example_ternary_tree.svg" width="720">
</p>

## Installation

`GPSO` package is tested and should run without any problems on python versions 3.6 and 3.7.

### One-liner
For those who want to optimise right away just
```bash
pip install pygpso
```
and go ahead! Make sure to check example notebooks in [the **examples** directory](examples/) to see how it works and what it can do. Or, alternatively, you can run interactive notebooks in binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jajcayn/pygpso.git/master?filepath=examples)

### Go proper
When you are the type of girl or guy who likes to install packages properly, start by cloning (or forking) this repository, then installing all the dependencies and finally install the package itself
```bash
git clone https://github.com/jajcayn/pygpso
cd pygpso/
pip install -r requirements.txt
# optionally, but recommended
pip install -r requirements_optional.txt
pip install .
```
Don't forget to test!
```bash
pytest tests/
```

## Usage
A guide on how to optimise and what can be done using this package is given as jupyter notebooks in [the **examples** directory](examples/). You can also try them out live thanks to binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jajcayn/pygpso.git/master?filepath=examples).

The basic idea is to initialise the parameter space in which the optimisation is to be run and then iteratively dig deeper and evaluate the objective function when necessary
```python
from gpso import ParameterSpace, GPSOptimiser


def objective_function(params):
    # params as a list or tuple
    x, y = params
    ...
    <some hardcore computation>
    ...
    return <float>

# bounds of the parameters we will optimise
x_bounds = [-3, 5]
y_bounds = [-3, 3]
space = ParameterSpace(parameter_names=["x", "y"], parameter_bounds=[x_bounds, y_bounds])
opt = GPSOptimiser(parameter_space=space, n_workers=4)
best_point = opt.run(objective_function)
```

The package also offers plotting functions for visualising the results. Again, those are documented and showcased in [the **examples** directory](examples/).

### Notes
Gaussian Processes regression uses normalised coordinates within the bounds [0, 1]. All normalisation and de-normalisation is done automatically, however when you want to call `predict_y` on GPR model, do not forget to pass normalised coordinates. The normalisation is handled by `sklearn.MinMaxScaler` and `ParameterSpace` instance offers a convenience functions for this: `ParameterSpace.normalise_coords(orig_coords)` and `ParameterSpace.denormalise_coords(normed_coords)`.

Plotting of the ternary tree (`gpso.plotting.plot_ternary_tree()`) requires `igraph` package, whose layout function is exploited. If you want to see the resulting beautiful tree, please install `python-igraph`.

Support of saver (for saving models run, e.g. timeseries along with the optimisation) is provided by `PyTables` (and `pandas` if you're saving results to `DataFrame`s).
 
## Known bugs and future improvements
* saving of GP surrogate is now hacky, as `GPFlow` not yet officially supports saving / loading of the models due to [bug in `tensorflow`](https://github.com/tensorflow/tensorflow/issues/34908). The hacky way, unfortunately, only supports basic kernels and mean functions, i.e. no kernel operations (such as sum or multiplication) allowed (for now).

## Final notes
When you encounter a bug or have any idea for an improvement, please open an issue and/or contact me.

When using this package in publications, please cite the original Jonathan's paper as
```bibtex
@article{hadida2018bayesian,
  title={Bayesian Optimisation of Large-Scale Biophysical Networks},
  author={Hadida, Jonathan and Sotiropoulos, Stamatios N and Abeysuriya, Romesh G and Woolrich, Mark W and Jbabdi, Saad},
  journal={Neuroimage},
  volume={174},
  pages={219--236},
  year={2018},
  publisher={Elsevier}
}
```
and acknowledge the usage of this package.
