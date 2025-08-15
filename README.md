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

Run `python -m mu_generator` to import the functions or see the source
for details.
