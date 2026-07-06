<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import wadler from './wadler.jl?raw';
	import julia from 'svelte-highlight/languages/julia';

	import { base } from '$app/paths';
</script>

<p id="top"><a href="{base}/">Back to list of talks</a></p>
{#snippet backToTop()}
	<a href="#top">Back to top</a>
{/snippet}

<h1 id="recording">Talk recording</h1>

<p>To come...</p>

<h1 id="links">Some other resources</h1>
<ul>
	<li>
		<a href="https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf"
			>The original paper (Wadler, 1999)</a
		>
	</li>
</ul>

<h2 id="wadler">Wadler's 'prettier printer', translated to Julia</h2>
{@render backToTop()}

<p>Please note that even though this is a word-for-word translation, it's not meant to have the same characteristics. In particular it's rather less efficient than the Haskell version because of the lack of laziness.</p>

<p>The idea in Haskell is that if you have a <code>Union</code> doc, and you pick the first option, the second option never even needs to be materialised. This matters especially when you have nested <code>Union</code> docs.</p>

<CodeExample anchorname={null} language={julia} filename="wadler.jl" code={wadler} />

<style>
	ul {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
</style>
