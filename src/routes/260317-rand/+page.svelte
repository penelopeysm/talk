<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import rand_py from './rand.py?raw';
	import crack from './crack.py?raw';
	import python from 'svelte-highlight/languages/python';

	import { base } from '$app/paths';
</script>

<p id="top"><a href="{base}/">Back to list of talks</a></p>
{#snippet backToTop()}
	<a href="#top">Back to top</a>
{/snippet}

<h1 id="recording">Talk recording</h1>

<p>The actual talk starts at around 4:40.</p>

<div id="_recording">
	<iframe
		width="840"
		height="473"
		src="https://www.youtube.com/embed/ZGtMeAg70Is?si=kIql_NfZHIqPyjp8"
		title="YouTube video player"
		frameborder="0"
		allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
		referrerpolicy="strict-origin-when-cross-origin"
		allowfullscreen
	></iframe>
</div>

<h1 id="links">Some other resources</h1>
<ul>
	<li>
		<a
			href="https://github.com/python/cpython/blob/40095d526bd8ddbabee0603c2b502ed6807e5f4d/Modules/_randommodule.c#L187-L193"
			>CPython 3.14's random float generator</a
		> (generate random int in [0, 2^53), then divide by 2^53)
	</li>
	<li>
		<a
			href="https://github.com/JuliaLang/julia/blob/6fa6f09451782c9a07c867fab6e878407dadef7b/stdlib/Random/src/generation.jl#L32-L35"
			>Julia 1.11's random float generator</a
		> (generate in [1, 2), then subtract 1)
	</li>

	<li>
		You might notice that we are not randomly generating all possible floats between 0 and 1. This
		algorithm is one way to get aroud it: <a href="https://specbranch.com/posts/fp-rand/"
			>Perfect Random Floating-Point Numbers</a
		>.
	</li>
	<li>
		<a href="https://www.smogon.com/ingame/rng/"
			>All you ever wanted to know about RNG in Pokémon games</a
		>
	</li>
	<li>
		Sometimes, it doesn't matter if your RNG isn't cryptographically secure (e.g., in simulations).
		But sometimes it does, and messing it up can have real consequences: see this example of an <a
			href="https://research.swtch.com/openssl">OpenSSL bug in random number generation</a
		> from 2008.
	</li>
</ul>

<h2 id="lcg">Linear congruential generators</h2>
{@render backToTop()}
<CodeExample anchorname={null} language={python} filename="rand.py" code={rand_py} />

<h2 id="crack">Reverse engineering an LCG</h2>
{@render backToTop()}
<CodeExample anchorname={null} language={python} filename="crack.py" code={crack} />

<style>
	ul {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
</style>
