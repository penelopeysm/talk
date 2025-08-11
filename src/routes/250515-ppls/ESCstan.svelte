<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import stan from 'svelte-highlight/languages/stan';

	const eight_schools_centered_stan = `
data {
  int<lower=0> J; // number of schools
  array[J] real y; // estimated treatment
  array[J] real<lower=0> sigma; // std of estimated effect
}
parameters {
  array[J] real theta; // treatment effect in school j
  real mu; // hyper-parameter of mean
  real<lower=0> tau; // hyper-parameter of sdv
}
model {
  tau ~ cauchy(0, 5); // a non-informative prior
  theta ~ normal(mu, tau);
  y ~ normal(theta, sigma);
  mu ~ normal(0, 5);
}`;
</script>

<CodeExample
	anchorname={null}
	language={stan}
	filename="eight_schools_centered.stan"
	code={eight_schools_centered_stan}
/>
