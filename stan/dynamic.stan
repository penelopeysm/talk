data {
  real y;
}
parameters {
  real x;
}
model {
  x ~ normal(0, 1);

  if (x > 0)
    y ~ normal(x, 1);
  else
    y ~ beta(2, 2);
}

generated quantities {
  real sigma;
  if (x > 0)
    sigma = 1;
  else
    sigma = 2;
}
