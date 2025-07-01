<script lang="ts">
	import Highlight from 'svelte-highlight';
	import stan from 'svelte-highlight/languages/stan';
	import python from 'svelte-highlight/languages/python';
	import julia from 'svelte-highlight/languages/julia';
	import atomOneLight from 'svelte-highlight/styles/atom-one-light';
    import { base } from '$app/paths';

	import CodeExample from '$lib/CodeExample.svelte';

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

	const eight_schools_centered_stan_py = `
from cmdstanpy import CmdStanModel
from pathlib import Path
import time

DATA = {
    "y": [28, 8, -3, 7, -1, 1, 18, 12],
    "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
    "J": 8,
}

def main():
    stan_file = Path(__file__).parent / "eight_schools_centered.stan"
    model = CmdStanModel(stan_file=stan_file)
    x = time.time()
    fit = model.sample(data=DATA, chains=1,
                       iter_warmup=10000, save_warmup=False,
                       iter_sampling=10000, thin=10)
    y = time.time()
    print(fit.summary())
    print(f"Time taken: {y - x} seconds")


if __name__ == "__main__":
    main()
`;

	const eight_schools_centered_pymc = `
import pymc as pm
import numpy as np
import time

J = 8
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

def main():
    with pm.Model() as eight_schools_centered:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu, tau, shape=J)
        obs = pm.Normal("obs", theta, sigma, observed=y)

        start = time.time()
        trace = pm.sample(draws=10000, tune=10000, chains=1)
        end = time.time()

    print(f"took {end - start} seconds")

if __name__ == "__main__":
    main()`;

	const eight_schools_centered_julia = `
using Turing

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function eight_schools_centered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(mu, tau)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

model_esc = eight_schools_centered(J, y, sigma)

chain = sample(model_esc, NUTS(), 1000; num_warmup=10000, thinning=10)
    `;

	const sampler_jl = `
using Turing, MicroCanonicalHMC

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function eight_schools_centered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(mu, tau)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

model_esc = eight_schools_centered(J, y, sigma)
sample(model_esc, NUTS(), 5000)

using MicroCanonicalHMC
mchmc_sampler = externalsampler(MCHMC(2000, 0.001))
sample(model_esc, mchmc_sampler, 5000)
    `;

	const submodel_jl = `
using Turing

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function priors()
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    return (mu=mu, tau=tau)
end

@model function eight_schools_centered(J, y, sigma)
    p ~ to_submodel(priors())
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(p.mu, p.tau)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

@model function eight_schools_noncentered(J, y, sigma)
    p ~ to_submodel(priors())
    theta_trans = Vector{Float64}(undef, J)
    for i in 1:J
        theta_trans[i] ~ Normal(0, 1)
        theta = theta_trans[i] * p.tau + p.mu
        y[i] ~ Normal(theta, sigma[i])
    end
end

model_priors = priors()
chain_priors = sample(model_priors, NUTS(), 2000)

model_esc = eight_schools_centered(J, y, sigma)
chain_esc = sample(model_esc, NUTS(), 2000)

model_esnc = eight_schools_noncentered(J, y, sigma)
chain_esnc = sample(model_esnc, NUTS(), 2000)</code></pre>
    `;

	const ad_jl = `
using Turing, ADTypes
import Enzyme: set_runtime_activity, Reverse

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function eight_schools_noncentered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta_trans = Vector{Float64}(undef, J)
    for i in 1:J
        theta_trans[i] ~ Normal(0, 1)
        theta = theta_trans[i] * tau + mu
        y[i] ~ Normal(theta, sigma[i])
    end
end

model = eight_schools_noncentered(J, y, sigma)

forwarddiff = AutoForwardDiff()
chain = sample(model, NUTS(; adtype=forwarddiff), 2000)

enzyme_reverse = AutoEnzyme(; mode=set_runtime_activity(Reverse, true))
chain = sample(model, NUTS(; adtype=enzyme_reverse), 2000)
    `;
</script>

<svelte:head>
	{@html atomOneLight}
</svelte:head>

<p><a href="{base}">Back to list of talks</a></p>

<h1 id="toc">Table of contents - code snippets</h1>
<ul>
	<li><a href="#escstan"><code>eight_schools_centered.stan</code></a></li>
	<li><a href="#escstanpy"><code>eight_schools_centered_stan.py</code></a></li>
	<li><a href="#escpymc"><code>eight_schools_centered_pymc.py</code></a></li>
	<li><a href="#escjulia"><code>eight_schools_centered.jl</code></a></li>
</ul>
<ul>
	<li><a href="#sampler"><code>sampler.jl</code></a></li>
	<li><a href="#submodel"><code>submodel.jl</code></a></li>
	<li><a href="#ad"><code>ad.jl</code></a></li>
</ul>

<br />
<hr />

<h1 id="links">Links</h1>
<p>Things mentioned in this talk:</p>
<ul>
	<li><a href="https://arxiv.org/pdf/2307.14339">MCHMC in astrophysics</a></li>
	<li>
		<a href="https://cdcgov.github.io/Rt-without-renewal/stable/showcase/replications/mishra-2020/"
			>Infectious disease modelling with EpiAware.jl</a
		>
	</li>
	<li><a href="https://arxiv.org/pdf/2505.05542">DifferentiationInterface.jl</a></li>
	<li><a href="https://turinglang.org/ADTests">Turing x AD table</a></li>
	<li><a href="https://enzyme.mit.edu/">Enzyme AD</a></li>
</ul>
<p>More generally:</p>
<ul>
	<li><a href="https://github.com/penelopeysm">Me on GitHub</a></li>
</ul>
<ul>
	<li><a href="https://turinglang.org/">Turing.jl docs</a></li>
	<li><a href="https://github.com/TuringLang">Turing.jl on GitHub</a></li>
</ul>
<ul>
	<li><a href="https://turing.ac.uk">The Alan Turing Institute</a></li>
	<li>
		<a href="https://alan-turing-institute.github.io/REG/"
			>Research Software Engineering at the Turing</a
		> <span class="small">(I made this website too)</span>
	</li>
</ul>

<br />
<hr />

<h1>Code snippets</h1>

{#snippet stan_desc()}
	<p>
		(This was taken from <a
			href="https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/eight_schools_centered.stan"
			>PosteriorDB</a
		>.)
	</p>
{/snippet}
<CodeExample
	anchorname="escstan"
	language={stan}
	filename="eight_schools_centered.stan"
	code={eight_schools_centered_stan}
	description={stan_desc}
/>

{#snippet stan_py_desc()}
	<p>
		You will need to have <a href="#escstan"><code>eight_schools_centered.stan</code></a> in the
		same directory as this file. Then create a virtual environment with the <code>cmdstanpy</code>
		package installed, and run (from the Python REPL) <code>cmdstanpy.install_cmdstan()</code> to set
		up Stan.
	</p>
{/snippet}
<CodeExample
	anchorname="escstanpy"
	language={python}
	filename="eight_schools_centered_stan.py"
	code={eight_schools_centered_stan_py}
	description={stan_py_desc}
/>

{#snippet pymc_desc()}
	<p>You will need a virtual environment with the <code>pymc</code> package installed.</p>
{/snippet}
<CodeExample
	anchorname="escpymc"
	language={python}
	filename="eight_schools_centered_pymc.py"
	code={eight_schools_centered_pymc}
	description={pymc_desc}
/>

{#snippet julia_desc()}
	<p>
		Launch Julia with <code>julia --project=.</code> and then in the Julia REPL, enter
		<code>]</code>. This will cause the REPL to enter package-manager mode, indicated by a prompt of
		<code>pkg&gt;</code>. Then type <code>add Turing</code> to install the Turing.jl package.
	</p>
	<p>
		Once that is done, press Backspace to re-enter the REPL mode, which has a prompt that looks like <code
			>julia&gt;</code
		>. You can then run the commands in this file.
	</p>
{/snippet}
<CodeExample
	anchorname="escjulia"
	language={julia}
	filename="eight_schools_centered.jl"
	code={eight_schools_centered_julia}
	description={julia_desc}
/>

{#snippet sampler_desc()}
	<p>
		Go back to the same directory that you ran the Julia code in. If you run <code
			>julia --project=.</code
		> it will activate the same environment as previously.
	</p>
	<p>
		As before, you will need to install a dependency: press <code>]</code>, then enter
		<code>add MicroCanonicalHMC</code>, then press Backspace. You can then run all of the code below
		in the REPL.
	</p>
{/snippet}
<CodeExample
	anchorname="sampler"
	language={julia}
	filename="sampler.jl"
	code={sampler_jl}
	description={sampler_desc}
/>

{#snippet submodel_desc()}{/snippet}
<CodeExample
	anchorname="submodel"
	language={julia}
	filename="submodel.jl"
	code={submodel_jl}
	description={submodel_desc}
/>

{#snippet ad_desc()}
	<p>Again, a few more dependencies: <code>]add ADTypes Enzyme</code></p>
{/snippet}
<CodeExample anchorname="ad" language={julia} filename="ad.jl" code={ad_jl} description={ad_desc} />
