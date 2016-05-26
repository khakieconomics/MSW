data {
  int T; // number of observations
  vector[T] Y; // my univariate time series
  real sigma_intercept;
  real sigma_beta;
}
parameters {
  // define parameters
  vector[T] intercept;
  vector[T] beta_1;
  real<lower = 0> sigma;// scale parameter must not be negative
  real<lower = 1> nu;// scale parameter must not be negative
}
model{
  // define priors
  intercept ~ normal(0, 1);
  beta_1 ~ normal(0, 1);
  sigma ~ cauchy(0, 2);
  nu ~ cauchy(7, 5);
  
  // priors for initial state
  intercept[1] ~ normal(5, 3);
  beta_1[1] ~ normal(1, 0.3);
  
  
  for(t in 2:T) {
    // define time varying parameter series
    intercept[t] ~ normal(intercept[t-1], sigma_intercept);
    beta_1[t] ~ normal(beta_1[t-1], sigma_beta);
    
    // define likelihood
    Y[t] ~ student_t(nu, intercept[t] + beta_1[t] * Y[t-1], sigma);
  }
  
  
}
generated quantities {
  vector[T] y_sim;
  vector[T] log_density;
  
  y_sim[1] <- 0;
  log_density[1] <- 0;
  
  for(t in 2:T) {
    y_sim[t] <- student_t_rng(nu, intercept[t] + beta_1[t] * Y[t-1], sigma);
    log_density[t] <- student_t_log(Y[t], nu, intercept + beta_1[t] * Y[t-1], sigma);
  }
}