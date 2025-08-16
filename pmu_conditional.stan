functions {
  real tail_logpdf(real x, real alpha, real mu0) {
    return alpha * log(mu0) - lgamma(alpha) - (alpha + 1) * log(x) - mu0 / x;
  }
  real lognormal_pdf(real x, real m, real s) {
    return exp(lognormal_lpdf(x | m, s));
  }
  real tail_pdf(real x, real alpha, real mu0) {
    return exp(tail_logpdf(x, alpha, mu0));
  }
}

data {
  int<lower=1> I;               // number of histogram conditions
  int<lower=1> J;               // number of mu bins
  vector[J] mu;                 // mu grid
  vector[J] dmu;                // bin widths in mu
  array[I, J] int<lower=0> cnt; // histogram counts
  vector<lower=0>[I] Ntot;      // total counts per condition
  int<lower=1> P;               // basis dimension
  matrix[I, P] B;               // basis matrix
  int<lower=1> K;               // number of lognormal components
  real<lower=0> sigma_min;      // lower bound for sigma
  real<lower=0> lambda_ent;     // negative entropy regulariser
  real<lower=0> lambda_tail;    // weak prior on tail exponent
  int<lower=0,upper=1> eq_weight; // weight datasets equally
}

parameters {
  vector[P] beta_m1;                // base mean
  matrix[P, K - 1] beta_m_diff;     // mean differences
  matrix[P, K] beta_logsig;         // log sigma params
  matrix[P, K + 1] beta_w;          // weight logits (incl. tail)
  vector[P] beta_alpha;             // tail exponent
  vector[P] beta_mu0;               // tail scale
}

transformed parameters {
  matrix[I, K] m;                  // means
  matrix[I, K] s;                  // sigmas
  matrix[I, K + 1] w;              // weights incl. tail
  vector[I] alpha;
  vector[I] mu0;

  for (i in 1:I) {
    row_vector[P] Bi = B[i];
    // means with ordering
    real m1 = Bi * beta_m1;
    m[i,1] = m1;
    for (k in 2:K) {
      real diff = Bi * beta_m_diff[, k - 1];
      m[i,k] = m[i,k-1] + log1p_exp(diff); // softplus difference
    }
    // sigmas
    for (k in 1:K) {
      real sr = Bi * beta_logsig[, k];
      s[i,k] = sigma_min + log1p_exp(sr);
    }
    // weights
    vector[K + 1] logits = (Bi * beta_w)';
    w[i] = (softmax(logits))';
    // tail params
    alpha[i] = 1 + log1p_exp(Bi * beta_alpha);
    mu0[i] = log1p_exp(Bi * beta_mu0);
  }
}

model {
  // priors on coefficients
  to_vector(beta_m1) ~ normal(0, 1);
  to_vector(beta_m_diff) ~ normal(0, 1);
  to_vector(beta_logsig) ~ normal(0, 1);
  to_vector(beta_w) ~ normal(0, 1);
  beta_alpha ~ normal(0, 1);
  beta_mu0 ~ normal(0, 1);

  for (i in 1:I) {
    real wgt = eq_weight ? inv(Ntot[i]) : 1.0;
    // negative entropy regularisation
    target += -lambda_ent * dot_product(w[i], log(w[i] + 1e-12));
    // weak prior on tail exponent
    target += -lambda_tail * (alpha[i] - 1);
    for (j in 1:J) {
      real pdf = 0;
      for (k in 1:K) {
        pdf += w[i,k] * lognormal_pdf(mu[j], m[i,k], s[i,k]);
      }
      pdf += w[i,K+1] * tail_pdf(mu[j], alpha[i], mu0[i]);
      real rate = fmax(1e-12, Ntot[i] * pdf * dmu[j]);
      target += wgt * poisson_lpmf(cnt[i,j] | rate);
    }
  }
}

generated quantities {
  // entropy for monitoring
  vector[I] mix_entropy;
  for (i in 1:I) {
    mix_entropy[i] = -dot_product(w[i], log(w[i] + 1e-12));
  }
}

