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

	const submodel_naive_jl = `
using Turing

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function eight_schools_noncentered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    # We have to change the name of this vector
    theta_trans = Vector{Float64}(undef, J)
    for i in 1:J
        # This was our original model:
        # theta[i] ~ Normal(mu, tau)
        # We changed it to:
        theta_trans[i] ~ Normal(0, 1)
        theta_i = theta_trans[i] * tau + mu
        # This likelihood term remains the same.
        y[i] ~ Normal(theta_i, sigma[i])
    end
end

model_esnc = eight_schools_noncentered(J, y, sigma)

chain = sample(model_esnc, NUTS(), 1000; num_warmup=10000, thinning=10)
ess(chain)
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

# ForwardDiff.jl is the default AD backend in Turing.jl.
# If you don't specify an AD backend, it will do the same as this.
forwarddiff = AutoForwardDiff()
chain = sample(model, NUTS(; adtype=forwarddiff), 2000)

# You can switch AD backends by passing a different adtype argument.
enzyme_reverse = AutoEnzyme(; mode=set_runtime_activity(Reverse, true))
chain = sample(model, NUTS(; adtype=enzyme_reverse), 2000)
    `;

	const manualAD = `
using Turing
import DynamicPPL, LogDensityProblems

@model f() = x ~ Normal()

function LogDensityProblems.logdensity_and_gradient(
    ldf::DynamicPPL.LogDensityFunction{<:DynamicPPL.Model{typeof(f)}},
    x::AbstractVector
)
    # This function must return the log probability density as the first
    # argument and the gradient as the second argument.
    # You could manually calculate the first argument too, but here we 
    # just defer to the existing logdensity function.
    return LogDensityProblems.logdensity(ldf, x), [-x[1]]
end

sample(f(), NUTS(), 1000)
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
	on 15 May 2025. The title that we publicly used was <i>'The Design of Turing.jl'</i>, mainly
	because the above title was too long :).
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

<h2 id="repro">Reproducibility</h2>

The Julia examples in this page were last tested with

<pre><code>
  [47edcb42] ADTypes v1.16.0
  [366bfd00] DynamicPPL v0.36.15
  [7da242da] Enzyme v0.13.66
  [6fdf6af0] LogDensityProblems v2.1.2
⌃ [234d2aa0] MicroCanonicalHMC v0.1.6
  [fce5fe82] Turing v0.39.10
</code></pre>

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
		<li><a href="#turing">(Some) unique Turing.jl features</a></li>
		<ul>
			<li><a href="#sampler">External samplers</a></li>
			<li><a href="#submodel">Submodels</a></li>
			<li><a href="#ad">Automatic differentiation</a></li>
			<li><a href="#andtherest">All the other stuff</a></li>
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
{#snippet empty()}{/snippet}
<CodeExample
	anchorname={null}
	language={julia}
	filename="eight_schools_centered.jl"
	code={eight_schools_centered_julia}
	description={empty}
/>

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
<CodeExample
	anchorname={null}
	language={python}
	filename="eight_schools_centered_stan.py"
	code={eight_schools_centered_stan_py}
	description={empty}
/>

Finally, we'll look at the same model in PyMC.

<div class="info">
	<h3>To run this...</h3>
	<p>You will need a virtual environment with the <code>pymc</code> package installed.</p>
</div>
<CodeExample
	anchorname={null}
	language={python}
	filename="eight_schools_centered_pymc.py"
	code={eight_schools_centered_pymc}
	description={empty}
/>

<div class="info">
	<h3>Julia's other PPLs</h3>
	<p>
		Julia also has other PPLs apart from Turing.jl, such as <a
			href="https://github.com/probcomp/Gen.jl"
			target="_blank">Gen.jl</a
		>
		and <a href="https://github.com/reactivebayes/RxInfer.jl" target="_blank">RxInfer.jl</a>. (I've
		probably missed out about 5 more.)
	</p>
	<p>
		Unfortunately I can't seem to access
		<a href="https://www.gen.dev/" target="_blank">the Gen.jl website</a>, so I shall refrain from
		fully commenting on it. My current impression of it (which seems supported by
		<a href="https://dl.acm.org/doi/pdf/10.1145/3314221.3314642" target="_blank">the paper</a>) is
		that it contains a huge library of tools for <i>programmable inference</i>, i.e., operations
		that tend to be common in Bayesian inference. In some ways this is more principled and also
		allows for an extreme level of customisation over inference algorithms. Turing on the other hand
		is much more like a 'framework' where you import a few functions, write a model, and sample with
		less fuss over how you're exactly doing it. (Note that Stan takes this to an extreme: the model
		is simply a function that returns a number and everything else stems from that.)
	</p>
	<p>
		RxInfer.jl has a rather different approach to modelling than Turing: it constructs a graph of
		all the variables in the model, which leads to some restrictions on what is possible in the
		model. In contrast Turing is more general-purpose and Turing models are genuine functions which
		can contain arbitrary Julia code. In return RxInfer can make use of this richer structure to
		perform inference more efficiently, for example when there are conjugate variables.
	</p>
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
	attention, and for good reason: people would like to get their results faster! model-dependent.
</p>

<p>
	<i>For this specific model</i>, we have that Stan &gt; Turing &gt; PyMC in terms of speed (that
	is, Stan is the fastest). You can verify this by running the code examples above yourself (helpful
	timing functions have been inserted into all of them). On my current laptop, I get: 0.7 seconds
	for Stan, 1.5 seconds for Turing, and 6 seconds for PyMC. However, the exact numbers can depend
	quite heavily on the model being tested as well as how they are implemented. For example, the
	Turing code can be sped up by using <code>MvNormal</code> instead of looping over several
	<code>Normal</code>s. I also don't pretend that I have given 100% optimised versions for the other
	PPLs.
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

<p>The consequence of this is twofold:</p>

<ul>
	<li>
		Firstly, to define a new probability distribution for use in Turing.jl, you don't have to figure
		out all of Turing's internals; you need only define something that obeys
		<a href="https://juliastats.org/Distributions.jl/stable/extends/" target="_blank"
			>Distributions.jl's interface</a
		>. Often this just entails writing Julia code; there is no need to dig into low-level languages
		(C++ for Stan) or numerical frameworks like Jax.
	</li>
	<li>
		Secondly, it means that any distributions that you define in this way can <i>also</i> be used in
		other packages that rely on Distributions.jl, which encourages code reuse and modularity.
	</li>
</ul>

<h2 id="turing">(Some) unique Turing.jl features</h2>
<a href="#toc">(Back to top)</a>

<p>
	The bulk of this section focuses on some things that Turing.jl can do precisely because the DSL is
	quite thin.
</p>

<h3 id="sampler">External samplers</h3>
<a href="#toc">(Back to top)</a>

<p>
	Much like adding a custom distribution, to add a custom sampler, you only need to implement an
	interface that is defined outside of Turing.jl. Specifically, any MCMC sampler needs to subtype <code
		>AbstractMCMC.AbstractSampler</code
	>, and define two <code>AbstractMCMC.step</code> methods that, respectively, define how to
	generate the first sample, and how to generate the <i>N</i>-th sample given the previous one.
</p>

<div class="warning">
	<b>Caveat:</b> this is actually not fully true, as of Turing v0.39. Although the main requirement
	is to fulfil the AbstractMCMC interface, there are some additional interface details that only
	come from Turing.jl, most notably <code>Turing.Inference.getparams</code>: see
	<a
		href="https://turinglang.org/Turing.jl/stable/api/Inference/#Turing.Inference.ExternalSampler"
		target="_blank">the Turing.jl documentation</a
	> for more information. These are likely to be removed in the near future.
</div>

<p>
	The relative ease of defining samplers makes Turing.jl a rich ground for implementing new
	inference techniques. For example, the microcanonical Hamiltonian Monte Carlo (MCHMC) sampler is
	implemented in the<a href="https://github.com/JaimeRZP/MicroCanonicalHMC.jl/" target="_blank"
		>MicroCanonicalHMC.jl</a
	>
	package. Using it in Turing.jl is as simple as wrapping it in a call to
	<code>externalsampler()</code>:
</p>

<div class="info">
	<h3>To run this...</h3>
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
</div>
<CodeExample
	anchorname={null}
	language={julia}
	filename="sampler.jl"
	code={sampler_jl}
	description={empty}
/>

<p>
	<a href="https://arxiv.org/pdf/2307.14339" target="_blank">Here is an example</a> of the MCHMC sampler
	being used in conjunction with Turing.jl models to perform inference on astrophysical data.
</p>

<h3 id="submodel">Submodels</h3>
<a href="#toc">(Back to top)</a>

<p>
	Because Turing.jl models are essentially pure Julia code (with the exception of <code>~</code> statements),
	code inside models can be refactored and shifted around much like one would expect to with ordinary
	functions.
</p>

<p>
	For example, consider an alternative ('non-centered') parameterisation of the eight-schools
	problem. Instead of writing that <code>theta[i] ~ Normal(mu, tau)</code>, we'll instead draw
	<code>theta_trans[i] ~ Normal(0, 1)</code>, before multipling it by <code>tau</code> and adding
	<code>mu</code>
	to get <code>theta</code>. This is an entirely equivalent model, but leads to better sampling
	especially when <code>tau</code> is small.
</p>

<p>
	If you run this code example, you should find that NUTS gives much better sampling results,
	especially for<code>tau</code> (you can see this from the effective sample size, which is
	calculated by <code>ess()</code>).
</p>

<CodeExample
	anchorname={null}
	language={julia}
	filename="submodel_naive.jl"
	code={submodel_naive_jl}
	description={empty}
/>

<p>
	However, writing out both this parameterisation and the centred one results in some repetition.
	For example, suppose we wanted to change the priors on <code>mu</code> and <code>tau</code>, and
	check whether there was still a difference between the two parameterisations: we'd have to change
	the priors in both models, which is not ideal.
</p>

<p>
	To avoid this, Turing.jl provides a <code>to_submodel</code> function that allows you to define a submodel
	that can be used in multiple models. This is very similar to how one would use a function in ordinary
	Julia code: you would extract out common sections into a single function. This code example demonstrates
	how:
</p>

<CodeExample
	anchorname={null}
	language={julia}
	filename="submodel.jl"
	code={submodel_jl}
	description={empty}
/>

<p>
	Notice how the priors are now defined only in the <code>priors()</code> model. In order for later
	models to 'see' the values of those parameters, we need to make sure to return them at the end of
	that submodel. Then, when we call <code>p ~ to_submodel(priors())</code>, we can access that
	return value as <code>p</code>.
</p>
<div class="warning">
	<b>Caveat:</b> The submodel interface may change in the future. In particular, we would like you
	to not have to explicitly return the parameters.
	<a href="https://github.com/TuringLang/Turing.jl/issues/2485" target="_blank">See this issue</a> for
	more information.
</div>

<p>
	Nested submodels also work as expected: this allows you to build up complex models from simple
	building blocks. This approach can be used, for example, in <a
		href="https://cdcgov.github.io/Rt-without-renewal/stable/showcase/replications/mishra-2020/"
		target="_blank">infectious disease modelling</a
	>
	In my opinion, the usefulness of this strategy has not been fully explored (nor has the design been
	fully optimised, as mentioned in the note above). It is a long-term aim of mine to build something
	similar to
	<a href="https://paulbuerkner.com/brms/" target="_blank"
		>the excellent <code>brms</code> package</a
	> using submodels.
</p>

<h3 id="ad">Automatic differentiation</h3>
<a href="#toc">(Back to top)</a>

<p>
	Gradient-based samplers, such as Hamiltonian Monte Carlo and its variants (like NUTS), require you
	to be able to evaluate the derivative of the model log probability density with respect to its
	parameters. This is usually accomplished using 'automatic differentiation', which does not
	calculate symbolic derivatives but rather provides you with the derivative of a function at a
	given point.
</p>

<p>
	There are a number of different approaches to automatic differentiation, and the Julia ecosystem
	has quite a few AD packages, with varying levels of maturity. True to its spirit, Turing.jl does
	not define its own AD backend, but rather makes use of... <i>all</i> of them! This is made
	possible by the <a href="https://github.com/SciML/ADTypes.jl" target="_blank">ADTypes.jl</a> and
	<a href=" https://github.com/JuliaDiff/DifferentiationInterface.jl/" target="_blank"
		>DifferentiationInterface.jl</a
	> packages, which provide a unified interface for interacting with AD backends.
</p>

<p>
	For example, <a href="https://enzyme.mit.edu/" target="_blank">Enzyme</a> is one of the most
	modern AD backends available in Julia. It works by differentiating code at the LLVM level, which
	generally leads to extremely high performance. Using Enzyme with Turing only requires one to add
	an <code>adtype</code> argument to the sampler:
</p>

<div class="warning">
	<b>Warning:</b> Enzyme compatibility with Turing to date has not always been guaranteed. As of the
	time of writing, Turing v0.39 (and DynamicPPL v0.36) do largely work with Enzyme! The most
	up-to-date status can be seen on
	<a href="https://turinglang.org/ADTests/" target="_blank">the ADTests page</a>.
</div>

<CodeExample anchorname={null} language={julia} filename="ad.jl" code={ad_jl} description={empty} />

<div class="info">
	<h3>What about Mooncake?</h3>
	<p>
		Mooncake is a wonderful AD package! It works very nicely with most Turing models, and the code
		quality is second to none.
	</p>
	<p>
		However, Mooncake's development has been quite closely tied to Turing's, at least in terms of
		the people working on it. In a traditional academic talk one might therefore like to highlight
		it. But I am not an academic, and my aim here is not to advertise my work but rather to make a
		point about the composability of the Julia ecosystem. I think using Enzyme as an example makes
		for an even stronger case.
	</p>
</div>

<p>
	The benefit of this is customisability and control. Different AD backends have different coverage
	of Julia features and performance characteristics, and this allows you to choose the one that
	works best for you.
</p>

<p>
	But how <i>do</i> you choose an AD backend? To help with this, we publish
	<a href="https://turinglang.org/ADTests/" target="_blank"
		>a table of performance benchmarks for AD backends on various Turing models</a
	>. Thus, you can see which of these your model might be similar to, and extrapolate from there.
	<a
		href="https://turinglang.org/docs/usage/automatic-differentiation/#choosing-an-ad-backend"
		target="_blank">Turing's documentation also provides some general tips</a
	>, and if you want to get more hands-on, it also describes how to benchmark AD backends yourself,
	so that you can tailor the choice to your specific model.
</p>

<div class="info">
	<h3>Manual derivatives</h3>
	<p>
		If you have analytic derivatives, you can specify this in a rather hacky way by overloading the
		<code>LogDensityProblems.logdensity_and_gradient</code> method. (Right now it is not possible to
		use DifferentiationInterface.jl for this.)
	</p>

	<p>
		In this model, there is a single parameter which is normally distributed. So the log probability
		density is <code>(const - x^2/2)</code>, and the gradient with respect to this parameter is
		<code>-x</code>. Because we take vectors as inputs and outputs, we need to index and wrap it
		back in a vector in the following implementation.

		<CodeExample
			anchorname={null}
			language={julia}
			filename="manual_ad.jl"
			code={manualAD}
			description={empty}
		/>
	</p>
</div>

<h3 id="andtherest">All the other stuff</h3>
<a href="#toc">(Back to top)</a>

<p>
	There are a ton of other integrations here which I don't have time to go into, like differential
	equations, neural networks, optimisation, Gaussian processes, MCMC tempering, and variational
	inference (including Pathfinder). The selection of topics covered here is not intended to imply
	that any of these are less important. Sorry if your favourite topic was not mentioned.
</p>

<h2 id="responsibility">What's difficult about this?</h2>
<a href="#toc">(Back to top)</a>

<p>
	So far I've presented a number of things that Turing.jl can do, by virtue of being a single
	package that fits into a larger Julia ecosystem. One might be prompted to ask if there is any <i
		>downside</i
	> to this.
</p>

<p>
	I think that the numerical computing ecosystem in Julia is quite unique and that this modularity
	(distributions defined in Distributions, etc.) is a particular strength of it. This does, however,
	also mean that there are a lot of moving parts and it's very possible for changes in one package
	to unwittingly break other packages. It also means that it's not clear whose responsibility it is
	to fix bugs or to run tests when some interaction between two packages goes wrong.
</p>

<p>
	Stan is a closed ecosystem, so if there's a bug in Stan's AD code, then the Stan developers know
	that it's their job to fix it. If a Turing model doesn't work with Enzyme, it's not immediately
	clear whose problem it is. So far things have worked out quite nicely: we try to create as minimal
	a working example as possible, usually one that does not involve Turing-specific code; then the
	Enzyme developers can fix the underlying bugs. This is quite a time-consuming process though (you
	can see <a href="https://github.com/TuringLang/ADTests/issues/3" target="_blank"
		>an example of me doing this here</a
	>) and it does somewhat rely on good faith on both sides to make it work.
</p>

<p>
	At the same time, it's a huge relief (to me) that I don't need to understand deeply how to write
	my own AD backend in order to maintain Turing.jl. In a way, the fact that Turing.jl can be
	developed by a relatively small developer team is only possible <i>because</i> we mostly only need
	to worry about the actual probabilistic programming bits. Of course, we have at least a passing knowledge
	of all the things we interact with, but we don't need to be experts.
</p>

<p>
	Another problem is that I've talked a lot about satisfying some <i>interface</i>, but interfaces
	in Julia are practically nonexistent. The only way to specify an interface is via documentation,
	and this has obvious problems: firstly things are often undocumented, and secondly even if they
	are, it's very easy for them to get out of date. (Note that Turing is hardly innocent in this
	regard:
	<a href="https://github.com/TuringLang/Turing.jl/pull/2640" target="_blank">some recent bugs</a>
	were introduced because the external sampler interface was not documented properly.)
</p>

<p>
	In general the only way to tell if you have satisfied an undocumented interface is to test out the
	code and see if it works. Of course this is a hugely unreliable way to develop software. There's a
	very nice
	<a href="https://invenia.github.io/blog/2020/11/06/interfacetesting/" target="_blank"
		>blog post by Invenia</a
	> describing this problem, along with and some strategies for overcoming it. I've long wanted to try
	to do this a bit better in Turing, though it's a long road ahead.
</p>

<p>
	Julia's multiple dispatch system, along with an (inexplicable) allergy towards modules, can also
	make bugs quite frustrating to track down: things may unexpectedly fail (or unexpectedly work
	correctly!) because of some function definition or import 5 files away.
</p>

<h2 id="choosing">Choosing a PPL</h2>
<a href="#toc">(Back to top)</a>

<p>
	With all of this said, we should probably return to the question of which PPL is 'the best'. This
	in itself is quite a loaded question. I hope that I've convinced you that Turing.jl has quite some
	cool features, but I also don't believe that that alone should mean that everybody should use
	Turing. Rather I think that it's important to understand the differences between the PPLs.
</p>

<p>
	In my opinion one of the biggest strengths of Turing is being able to write plain Julia code in
	your model, and also an extreme level of customisability. Thus if you want to be really able to
	control what goes on, Turing is a really good choice, and this is especially so if you are
	experimenting with new inference techniques. (In many ways this is really a strength of Julia,
	which we inherit by virtue of being a Julia package.)
</p>

<p>
	It is not true that every user wants to be able to control their model in this way. If you simply
	want to have top-notch performance and you don't care about how your AD is being performed, then
	Stan provides a perfectly good solution with their in-house (reverse-mode) AD. Stan's
	documentation, and the amount of existing (and up-to-date!) tutorials out there is far greater
	than for Turing. In many ways Turing's documentation assumes that you already know what Bayesian
	modelling is, and it teaches you <i>how</i> to do it in Turing.jl, but this can really be quite unapproachable
	for beginners.
</p>

<p>
	There are a huge number of reasons why one might choose a PPL over another that have absolutely
	nothing to do with technical design choices. A huge one is of course the language that it is
	written in. I think it is a not uncommon experience in the Julia community to advertise a library
	to someone else, only to get the response that 'but then I'd have to use Julia'. If you want to
	write your modelling code in pure(ish) Python, then PyMC lets you do this; if you don't mind
	calling an external Stan model, then all your modelling can be written in Python, and this would
	let you combine it with (for example) data processing in Python, without having to engineer a
	multi-language codebase.
</p>

<h2 id="support">How you can support your favourite PPL</h2>
<a href="#toc">(Back to top)</a>

<p>
	Regardless of which PPL you choose to use, I would love to encourage you to support it (and
	open-source software)!
</p>

<ol>
	<li>
		<b>Use it</b>: Open-source software generally does not pay bills. I'm very lucky to be able to
		work full-time on Turing.jl (for now), but at the very least, we need to demonstrate that people
		use what we write.
	</li>

	<li>
		<b>Cite papers</b>: Turing.jl is primarily an academic project so this really helps. You can
		find a list of papers in
		<a
			href="https://github.com/TuringLang/Turing.jl?tab=readme-ov-file#citing-turingjl"
			target="_blank">the Turing.jl README</a
		>.
	</li>

	<li>
		<b>Get in touch</b>: In order for us to understand what the most important issues are, we do
		need to have bidirectional communication with the community. Please don't hesitate to open an
		issue even if it's just to discuss something: we're quite friendly (I think). I've been trying
		to put together an (approximately...) fortnightly newsletter for Turing.jl, which is posted on
		<a href="https://julialang.slack.com/archives/CCYDC34A0" target="_blank">Slack</a>,
		<a href="https://github.com/TuringLang/Turing.jl/issues/2498" target="_blank">GitHub</a>, and
		<a href="https://turinglang.org/news" target="_blank">our website</a>.
	</li>

	<li>
		<b>Contribute code (?!)</b>: If you would like to get involved, we would love to have you on
		board as well. It can be quite difficult to navigate the internals of Turing but I am definitely
		more than happy to help guide new contributors. Of course, everyone is busy, so we hardly expect
		this. But <i>if</i> you do think that you might want to have a job in the future writing software,
		it's very impressive to put on your CV that you've contributed to a big open-source project :)
	</li>
</ol>

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
		margin: 20px;
		border-radius: 10px;
		padding: 10px 20px;
		background-color: #e0f7fa;
		font-size: 0.9em;

		h3 {
			margin: 0;
		}

		:last-child {
			margin-bottom: 0;
		}
	}

	div.warning {
		margin: 0 20px 20px 20px;
		border-radius: 10px;
		padding: 10px 20px;
		font-size: 0.9em;
		background-color: #f9d4e2;
		:last-child {
			margin-bottom: 0;
		}
	}
</style>
