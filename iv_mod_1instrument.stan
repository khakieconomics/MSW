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
  depvars = append_col(X_endog, Y);
}
parameters {
  matrix[2, P+1] beta;
  corr_matrix[2] omega;
  vector<lower = 0>[2] tau;
  vector[2] alpha;
}
transformed parameters {
  matrix[N, 2] depvars_hat;
  
  for(i in 1:N) {
    depvars_hat[i,1] = Z1[i]*beta[1]';
    depvars_hat[i,2] = X[i]*beta[2]'[1:(cols(beta[2])-1)] + depvars_hat[i,1]*beta[2][cols(beta[2])];
  }
}
model {
  // Priors
  for(i in 1:2) {
    beta[i] ~ normal(0, 1);
  }
  alpha ~ normal(0, 1);
  omega ~ lkj_corr(4);
  tau ~ student_t(5, 0, 2);
  
  for(i in 1:N){
    to_vector(depvars[i]) ~ multi_normal(alpha + to_vector(depvars_hat[i]), quad_form_diag(omega, tau));
  }
  
  
}