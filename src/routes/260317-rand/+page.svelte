<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import python from 'svelte-highlight/languages/python';

	import { base } from '$app/paths';
</script>

<p id="top"><a href="{base}/">Back to list of talks</a></p>
{#snippet backToTop()}
	<a href="#top">Back to top</a>
{/snippet}

<h1 id="links">Some other resources</h1>
<ul>
    <li><a href="https://github.com/python/cpython/blob/40095d526bd8ddbabee0603c2b502ed6807e5f4d/Modules/_randommodule.c#L187-L193">CPython 3.14's random float generator</a></li>
    <li><a href="https://research.swtch.com/openssl">OpenSSL's bug in random number generation</a>: what happens IRL when you mess up your RNG</li>
    <li><a href="https://specbranch.com/posts/fp-rand/">Perfect Random Floating-Point Numbers</a>: how to get around the incompleteness of float rand</li>
    <li><a href="https://www.smogon.com/ingame/rng/">All you ever wanted to know about RNG in Pokémon games</a></li>
</ul>

<!-- <h2 id="lcg">Linear congruential generators</h2> -->
<!-- {@render backToTop()} -->
<!-- <CodeExample anchorname={null} language={julia} filename="lcg.jl" code={lcg_jl} /> -->
<!---->
<!-- <h2 id="fp8">Floating point representation</h2> -->
<!-- {@render backToTop()} -->
<!-- <CodeExample anchorname={null} language={julia} filename="fp8.jl" code={fp8_jl} /> -->

<style>
	ul {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
</style>
