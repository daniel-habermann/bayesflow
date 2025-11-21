data {
  int<lower=1> J;
  int<lower=1> N;
  matrix[J, N] y;
}

parameters {
  real hyper_mean;
  real<lower=0> hyper_std;
  real<lower=0> shared_std;
  vector<offset=hyper_mean, multiplier=hyper_std>[J] local_mean;
}

model {
  hyper_mean ~ std_normal();
  hyper_std ~ std_normal();
  shared_std ~ std_normal();

  for (j in 1:J) {
    to_row_vector(y[j]) ~ normal(local_mean[j], shared_std);
  }
}
