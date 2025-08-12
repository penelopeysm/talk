<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import julia from 'svelte-highlight/languages/julia';

	import { base } from '$app/paths';
	import sir_jl from './sir.jl?raw';
</script>

<p><a href="{base}/">Back to list of talks</a></p>

<!-- <h1 id="toc">Table of contents</h1> -->

<h1 id="links">Some other resources</h1>
<ul>
	<li>
		<a href="https://sbfnk.github.io/mfiidd/pmcmc.html" target="_blank"
			>Model fitting and inference for infectious disease dynamics (Funk <i>et al.</i>)</a
		> I found this somewhat helpful for understanding how SMC worked (especially the visualisations in
		the lecture slides). This does not cover particle MCMC.
	</li>
	<li>
		<a href="https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf" target="_blank"
			>The infamous pMCMC review by Doucet <i>et al.</i></a
		> which offers a very thorough (and dense) theoretical justification for why particle MCMC methods
		work. I've spent a reasonable amount of time attempting to read it, but with very little success.
	</li>
	<li>
		A rather more friendly exposition of particle MCMC can be found on Darren Wilkinson's blog: <a
			href="https://darrenjw.wordpress.com/2011/05/17/the-particle-marginal-metropolis-hastings-pmmh-particle-mcmc-algorithm/"
			target="_blank">post 1 on PMMH</a
		>,
		<a
			href="https://darrenjw.wordpress.com/2014/01/25/introduction-to-the-particle-gibbs-sampler/"
			target="_blank">post 2 on PG</a
		>.
	</li>
</ul>
<p>Truthfully though I worked most of this out with ChatGPT.</p>

<h2 id="sir">Sampling–importance–resampling</h2>
<CodeExample anchorname={null} language={julia} filename="sir.jl" code={sir_jl} />

<style>
	ul {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
</style>
