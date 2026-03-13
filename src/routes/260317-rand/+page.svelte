<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import julia from 'svelte-highlight/languages/julia';

	import { base } from '$app/paths';
	import lcg_jl from './lcg.jl?raw';
	import fp8_jl from './fp8.jl?raw';
</script>

<p id="top"><a href="{base}/">Back to list of talks</a></p>
{#snippet backToTop()}
	<a href="#top">Back to top</a>
{/snippet}

<h1 id="links">Some other resources</h1>
<ul>
    <li>...</li>
</ul>

<h2 id="lcg">Linear congruential generators</h2>
{@render backToTop()}
<CodeExample anchorname={null} language={julia} filename="lcg.jl" code={lcg_jl} />

<h2 id="fp8">Floating point representation</h2>
{@render backToTop()}
<CodeExample anchorname={null} language={julia} filename="fp8.jl" code={fp8_jl} />

<style>
	ul {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
</style/
