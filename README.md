Event-Based Model
=================

The Event-Based Model (EBM) is a simple, robust model for the estimation of the most likely order of events in a degenerative disease. This version uses nonparametric distributions within the mixture modelling step.

Important Links
===============

- [Issue tracker](https://github.com/ucl-pond/kde_ebm/issues)

KDE EBM paper
-------------
- [Firth, *et al.*, bioRxiv, **2018**](https://doi.org/10.1101/297978)

EBM Papers
----------
- [Young *et al.*, Brain, **2014**](http://brain.oxfordjournals.org/cgi/pmidlookup?view=long&pmid=25012224)
- [Fonteijn *et al.*, NeuroImage, **2012**](http://www.sciencedirect.com/science/article/pii/S1053811912000791)

Dependencies
============
- [NumPy](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)

The code depends heavily on NumPy, uses SciPy to calculate some stats and do some optimisation and uses Matplotlib just to do the plotting.

Contributing
============
Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file before making any contributions.

Tweaks
======

Implemented in variable_bandwidth and neils_tweaks branches

1. Variable bandwidth KDE

   Improved mixture model fitting in low-density areas such as the tail of a skewed distribution
  
2. Prioritised controls

   Option for the controls to be 'fixed', i.e., very unlikely to be relabelled as 'abnormal' in the mixture model
