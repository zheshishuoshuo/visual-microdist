# visual-microdist

Tools for working with histogram magnification data and constructing a
continuous generator `p(mu | kappa, gamma, s)`.

The new module `mu_generator.py` implements:

* loading cached histograms and interpolating them onto a common grid,
* fitting each histogram with a lognormal + power-law mixture via Poisson
  likelihood minimisation,
* building an RBF interpolator to map `(kappa, gamma, s)` to the fitted
  parameters `psi`,
* evaluation and sampling utilities for the resulting analytic
  distribution.

In addition to being importable as a module, `mu_generator.py` provides a
small command line interface that can draw sample magnifications.  For
example:

```
python mu_generator.py --kappa 0.5 --gamma 0.5 --s 0.5 --size 3
```

This builds a lightweight model from the cached histograms and prints
three random `mu` samples.
