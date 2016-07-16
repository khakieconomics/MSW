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
  matrix[N, 2] depvars_hat;
  
  for(i in 1:N) {
    depvars_hat[i,1] = Z1*beta[1];
    depvars_hat[i,2] = X*beta[2][1:(rows(beta[2])-1)] + depvars_hat[i,1]*beta[2][rows(beta[2])];
  }
}
model {
  // Priors
  for(i in 1:2) {
    beta[i] ~ normal(0, 1);
  }
  omega ~ lkj_corr(4);
  tau ~ student_t(5, 0, 2);
  
  depvars ~ multi_normal(depvars_hat, quad_form_diag(omega, tau))
  
}