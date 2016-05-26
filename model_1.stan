data {
  int T; // number of observations
  vector[T] Y; // my univariate time series
}
parameters {
  // define parameters
  real intercept;
  real beta_1;
  real beta_2;
  real<lower = 0> sigma;// scale parameter must not be negative
}
model{
  // define priors
  intercept ~ normal(0, 1);
  beta_1 ~ normal(0, 1);
  beta_2 ~ normal(0, 1);
  sigma ~ cauchy(0, 2);
  
  // define likelihood
  for(t in 3:T) {
    Y[t] ~ normal(intercept + beta_1 * Y[t-1] + beta_2 * Y[t-2], sigma);
  }
}
generated quantities {
  vector[T] y_sim;
  vector[T] log_density;
  
  
  for(t in 1:2) {
    y_sim[t] <- 0;
    log_density[t] <- 0;
  }
  
  for(t in 3:T) {
    y_sim[t] <- normal_rng(intercept + beta_1 * Y[t-1] + beta_2 * Y[t-2], sigma);
    log_density[t] <- normal_log(Y[t], intercept + beta_1 * Y[t-1] + beta_2 * Y[t-2], sigma);
  }
}