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
from cmdstanpy import CmdStanModel, install_cmdstan
from pathlib import Path
import time

install_cmdstan()

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

	const badmodel = `
using Turing

@model function f(x)
    y = x + 1
    y ~ Normal()
end

sample(f(1.0), NUTS(), 1000)
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

<p><a href={base}>Back to list of talks</a></p>

<h1>Why Turing.jl? A limited comparison of probabilistic programming frameworks</h1>

<p>
	This talk was given at <a href="https://learnbayes.se/events/design-of-turing-jl/" target="_blank"
		>the Learn Bayes seminar series at Karolinska Institutet</a
	>
	on 15 May 2025. The original title was <i>'The Design of Turing.jl'</i>. I'm not fully happy with
	that title, but it was mostly due to length :).
</p>

<p>
	I later gave a reprise at <a
		href="https://github.com/alan-turing-institute/probabilistic-programming-study-group"
		target="_blank">The Alan Turing Institute's probabilistic programming study group</a
	> on 4 June 2025.
</p>

<p>
	For personal reasons, the recording of this talk is not publicly available. (If you are at the
	ATI, you can access the study group recording.) Thus I have endeavoured to provide a full text
	writeup of the talk here.
</p>

<p>
	As the title suggests, this talk involves <i>some</i> comparison of Turing.jl with other PPLs, specifically
	Stan and PyMC. However, the main focus of the talk is really about the unique features of Turing.jl
	and how they arise from the design of the library and other technical aspects of Julia.
</p>

<h2 id="toc">Table of contents</h2>

<ul class="toplevel">
	<div>
		<li><a href="#esc">'Eight schools' in various PPLs</a></li>
		<ul>
			<li><a href="#syntax">Comparing PPL syntaxes</a></li>
			<li><a href="#speed">Comparing PPL speeds</a></li>
		</ul>
	</div>
	<li><a href="#thesis">The central thesis of this talk</a></li>
	<div>
		<li><a href="#turing">Unique Turing.jl features</a></li>
		<ul>
			<li><a href="#sampler">External samplers</a></li>
			<li><a href="#submodel">Submodels</a></li>
			<li><a href="#ad">Automatic differentiation</a></li>
		</ul>
	</div>
	<li><a href="#responsibility">What's difficult about this?</a></li>
	<li><a href="#choosing">Choosing a PPL</a></li>
	<li><a href="#support">How you can support your favourite PPL</a></li>
</ul>

<h2 id="esc">'Eight schools' in various PPLs</h2>
<a href="#toc">(Back to top)</a>

<p>
	The 'eight schools' problem (<a href="https://www.jstor.org/stable/1164617" target="_blank"
		>Rubin, 1981</a
	>) is a classic example of a Bayesian model (specifically, a hierarchical model). In this model,
	students at eight different schools were given coaching for the SAT examination, and the average
	effect on their scores (<code>y</code>) was measured for each school, along with the standard
	deviation (<code>sigma</code>).
</p>

<p>
	The <a href="https://www.tensorflow.org/probability/examples/Eight_Schools" target="_blank"
		>TensorFlow Probability website</a
	> has more detailed information on the model. Here we provide only a short overview:
</p>

<ul>
	<li>
		<code>mu</code> and <code>tau</code> represent respectively, the 'true' mean effect that is common
		to all schools, and the standard deviation of this (i.e., how much it can vary from school to school).
	</li>
	<li>
		<code>theta[i]</code> represents the true underlying effect on school <i>i</i>. It is assumed to
		be normally distributed with mean <code>mu</code> and standard deviation <code>tau</code>.
	</li>
	<li>
		<code>y[i]</code>, which is the observed effect on school <i>i</i>, is assumed to be normally
		distributed with mean <code>theta[i]</code> and standard deviation
		<code>sigma[i]</code>.
	</li>
</ul>

<p>
	In this way we have allowed for each of the schools to have a different effect (<code
		class="inline">theta[i]</code
	>), while also sharing a common source via <code>mu</code> and
	<code>tau</code>.
</p>

<p>
	When taking a Bayesian approach to this modelling, we need to set some kind of prior for the
	parameters <code>mu</code> and <code>tau</code>. Here, we choose fairly non-informative priors:
	<code>mu</code>
	is given a normal distribution with a large standard deviation of 5, and <code>tau</code> follows a
	Cauchy distribution, but truncated to be non-negative (i.e., it cannot be less than 0).
</p>

<p>Putting all of this together, we get a Turing.jl model:</p>

{#snippet empty()}{/snippet}
<CodeExample
	anchorname={null}
	language={julia}
	filename="eight_schools_centered.jl"
	code={eight_schools_centered_julia}
	description={empty}
/>

<div class="info">
	<h3>To run this...</h3>
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
</div>

<p>
	The exact same model can be written in Stan as follows (with credit to <a
		href="https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/eight_schools_centered.stan"
		>PosteriorDB</a
	>):
</p>

{#snippet stan_desc()}{/snippet}
<CodeExample
	anchorname={null}
	language={stan}
	filename="eight_schools_centered.stan"
	code={eight_schools_centered_stan}
	description={stan_desc}
/>

<p>
	To sample from this model, you'll have to go via Python, R, or Julia. Here is a Python script that
	executes the model above.
</p>

<CodeExample
	anchorname={null}
	language={python}
	filename="eight_schools_centered_stan.py"
	code={eight_schools_centered_stan_py}
	description={empty}
/>

<div class="info">
	<h3>To run this...</h3>
	<p>
		You will need to have the Stan model and this Python script saved in the same directory. The
		Stan model should be named <code>eight_schools_centered.stan</code>.
	</p>
	<p>
		Then create a virtual environment with the <code>cmdstanpy</code> package installed. Inside this
		virtual environment, you can run the script with
		<code>python eight_schools_centered_stan.py</code>.
	</p>
</div>

Finally, we'll look at the same model in PyMC.

<CodeExample
	anchorname={null}
	language={python}
	filename="eight_schools_centered_pymc.py"
	code={eight_schools_centered_pymc}
	description={empty}
/>
<div class="info">
	<h3>To run this...</h3>
	<p>You will need a virtual environment with the <code>pymc</code> package installed.</p>
</div>

<h3 id="syntax">Comparing PPL syntaxes</h3>
<a href="#toc">(Back to top)</a>

<p>
	Between the three PPLs, Turing.jl arguably has the 'nicest' syntax: writing a Turing.jl model is
	almost like writing the mathematical equations that define the model. PyMC has a fairly similar
	syntax although it requires a little bit more boilerplate (for example, parameter names get
	specified twice, once as a variable to assign to and once as a string for PyMC to keep track of).
</p>
<p>
	Notably, both PyMC and Turing allow you to define parameters 'as you go along'. This is rather
	like a dynamically typed programming language. On the other hand, Stan requires you to separately
	define a list of parameters and their sizes before they can be used in the model. This in turn
	resembles a statically typed programming language.
</p>

<p>
	It would not be unfair to say that Stan is more verbose. However, much like how both dynamic and
	static languages exist, there <i>are</i> benefits to being explicit sometimes.
</p>
<p>
	For example, it makes it very clear which quantities are parameters and which ones are observed.
	In Turing.jl, data variables are specified in one of two ways: either by listing them as model
	arguments, or via an explicit conditioning syntax. However, there can be some ambiguity over this.
	For example, the following model (<a
		href="https://github.com/TuringLang/DynamicPPL.jl/issues/519"
		target="_blank">GitHub issue</a
	>) does not have well-defined behaviour:
</p>

<CodeExample
	anchorname={null}
	language={julia}
	filename="badmodel.py"
	code={badmodel}
	description={empty}
/>

<p>
	It appears here that <code>x</code> is a data variable, since it is a model argument. Thus,
	<code>y</code>
	is entirely derived from data. However, actually performing the sampling, one will find that
	<code>y</code> is treated as a parameter.
</p>

<h3 id="speed">Comparing PPL speeds</h3>
<a href="#toc">(Back to top)</a>

<p>
	Speed is one of the topics which I would prefer to not go into. It always garners a lot of
	attention, and for good reason: people would like to get their results faster!
</p>

<p>
	<i>In general</i> I would make the claim that Stan &gt; Turing &gt; PyMC in terms of speed (that is,
	Stan is the fastest). You can verify this by running the code examples above yourself (helpful timing
	functions have been inserted into all of them). On my current laptop, I get: 0.7 seconds for Stan,
	1.5 seconds for Turing, and 6 seconds for PyMC.
</p>

<p>
	Note that the <i>first</i> time you sample from a Turing model, it will be quite slow. This is mainly
	because of Julia's just-in-time compilation model: code is not compiled until it is called. (This is
	the origin of one of the most common complaints about Julia, the 'time to first X': it takes a long
	time to run anything for the first time.)
</p>

<p>
	Stan has a similar 'feature' as the Stan model has to be compiled before it can be executed. In
	fact, you may well have found similar behaviour for Python in general, even though CPython is not
	a just-in-time implementation: this is again to do with <a
		href="https://nedbatchelder.com/blog/201803/is_python_interpreted_or_compiled_yes.html"
		target="_blank">code compilation</a
	>.
</p>

<h2 id="thesis">The central thesis of this talk</h2>
<a href="#toc">(Back to top)</a>

<p>
	The main difference that I want to highlight, though, is that Turing.jl has <b
		>a very thin domain-specific layer</b
	>. That is to say, the amount of Turing.jl-specific code that a Turing user writes is quite low,
	essentially limited to <code>@model</code> and <code>~</code>.
</p>

<p>
	For example, in a Turing model, we have to specify a number of different distributions such as <code
		>Normal(mean, sd)</code
	>. These distributions are <i>not</i> defined in Turing.jl, but form part of a much more general
	package,
	<a href="https://github.com/JuliaStats/Distributions.jl" target="_blank">Distributions.jl</a>.
</p>

<p>
	In contrast, PyMC's distributions don't exist as distributions in their own right: they have a
	'special' meaning in that instantiating them adds them to the model. That is why you have to
	import them from <code>pymc</code>. To obtain an object that behaves like a 'distribution', you
	need to call (for example) <code>pm.Normal.dist(mean, sd)</code>.
</p>

<p>
	Stan, of course, exists as a closed ecosystem: its distributions are defined directly in the Stan
	codebase and you cannot add new distributions from outside. (Indeed you do not really define new
	distributions <i>per se</i>; this can instead be accomplished by
	<a href="https://mc-stan.org/docs/stan-users-guide/custom-probability.html" target="_blank"
		>adding custom probability terms</a
	>.)
</p>

<p>
	The consequence of this is twofold: firstly, to define a new probability distribution for use in
	Turing.jl, you don't have to figure out all of Turing's internals; you need only define something
	that obeys
	<a href="https://juliastats.org/Distributions.jl/stable/extends/" target="_blank"
		>Distributions.jl's interface</a
	>. Secondly, it means that any distributions that you define in this way can <i>also</i> be used in
	other packages that rely on Distributions.jl, which encourages code reuse and modularity.
</p>

<!--

<h2 id="links">Links</h2>
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

<h2>Code snippets</h2>


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


-->

<style>
	ul.toplevel > * {
		margin-bottom: 0.7em;
	}

	h2 {
		margin-top: 40px;
	}
	h3 {
		margin-bottom: 0;
	}
	div.info {
		margin: 0 20px 20px 20px;
		padding: 10px 20px;
		background-color: #e0f7fa;

		h3 {
			margin: 0;
		}

		:last-child {
			margin-bottom: 0;
		}
	}
</style>
