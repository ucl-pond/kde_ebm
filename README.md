[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ucl-pond/kde_ebm/HEAD?urlpath=lab/tree/examples)

KDE Event-Based Model
=================

The Event-Based Model (EBM) is a simple, robust model for the estimation of the most likely order of events in a degenerative disease. This version uses nonparametric (kernel density estimate) distributions within the mixture modelling step.

Installation instructions: see [INSTALL](INSTALL.md) file.

You can now try out the walkthrough notebook on [Binder](https://mybinder.readthedocs.io/en/latest/introduction.html). Just click the Launch Binder badge above. 

Important Links
===============

- [Issue tracker](https://github.com/ucl-pond/kde_ebm/issues)

KDE EBM paper
-------------
- [Firth, *et al.*, Alzheimer's & Dementia, **2020**](https://doi.org/10.1002/alz.12083)

EBM Papers
----------
- [Fonteijn *et al.*, NeuroImage, **2012**](https://doi.org/10.1016/j.neuroimage.2012.01.062)
- [Young *et al.*, Brain, **2014**](https://doi.org/10.1093/brain/awu176)

Dependencies
============
See [INSTALL](INSTALL.md) for installation instructions.

- [NumPy](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [awkde](https://github.com/noxtoby/awkde)

The code depends heavily on NumPy, uses SciPy to calculate some stats and do some optimisation and uses Matplotlib just to do the plotting. awkde is for variable bandwidth KDE (meged into main/master branch in Feb 2020).

Contributing
============
Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file before making any contributions.

Tweaks
======

1. Variable bandwidth KDE ([awkde](https://github.com/noxtoby/awkde)) mixture modelling

   Improved mixture model fitting in low-density areas such as the tail of a skewed distribution
  
2. Prioritised controls

   Option for the controls to be 'fixed', i.e., very unlikely to be relabelled as 'abnormal' in the mixture model
