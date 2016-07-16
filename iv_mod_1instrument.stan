data {
  int N; // number of observations
  int P; // number of non-endogenous covariates
  vector[N] X_endog; // Your endogenous regressor
  matrix[N, P] X; // Your non-endogenous regressors
  vector[N] Z; // Your instrument
  vector[N] Y;// The outcome
}
transformed data {
  matrix[N, P+1] Z1;
  matrix[N, 2] depvars;
  Z1 = append_col(X, Z);
}
parameters {
  vector[P+1] beta[2];
  corr_matrix[2] omega;
  vector<lower = 0> tau[2];
}
transformed parameters {
  vector[N] X_endog_fitted;
  
  X_endog_fitted = Z1*beta[1]
}
model {
  // Priors
  for(i in 1:2) {
    beta[i] ~ normal(0, 1);
  }
  omega ~ lkj_corr(4);
  tau ~ student_t(5, 0, 2);
  
  depvars ~ multi_normal(, quad_form_diag(omega, tau))
  
}