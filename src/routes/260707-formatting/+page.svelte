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
			>The original paper (Wadler, 1999).</a
		>
		Code formatters that use this approach include, notably, JavaScript's <code>prettier</code> (see
		<a href="https://prettier.io/docs/technical-details">their 'technical details' page</a> for more info).
	</li>
	<li>
		<a href="https://lindig.github.io/papers/strictly-pretty-2000.pdf"
			>Strictly Pretty (Lindig, 2000)</a
		>, a follow-up that adapts Wadler's approach to a strict language (the paper uses OCaml).
	</li>
	<li>
		The predominant other approach seems to be to use a cost function model where one seeks to find
		the layout with the lowest cost. Some papers include:
		<ul>
			<li>
				<a
					href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44667.pdf"
					><i>A New Approach to Optimal Code Formatting</i> (Yelland, 2016)</a
				>
			</li>
			<li>
				<a href="https://doi.org/10.1145/3110250"
					><i>A Pretty But Not Greedy Printer</i> (Bernardy, 2017)</a
				>
			</li>
			<li>
				<a href="https://doi.org/10.1145/3622837"
					><i>A Pretty Expressive Printer</i> (Porncharoenwase <i>et al.</i>, 2023)</a
				>
			</li>
		</ul>
		And some formatters that use this approach include<a
			href="https://github.com/llvm/llvm-project/blob/6e0311f077601350d5cc70e59c49432d985ca631/clang/lib/Format/Format.cpp#L2043-L2054"
			><code>clang-format</code></a
		>
		and
		<a href="https://journal.stuffwithstuff.com/2015/09/08/the-hardest-program-ive-ever-written/"
			><code>dartfmt</code></a
		>.
	</li>
	<li>
		There's some conceptual similarity between this and line-breaking algorithms for prose (i.e.,
		pretty-printing <i>paragraphs</i>). It's very easy to do a greedy algorithm (does this word fit
		on this line? If so, print it, else move to the next line) but TeX, for example, has
		<a href="https://en.wikipedia.org/wiki/Knuth%E2%80%93Plass_line-breaking_algorithm"
			>an algorithm</a
		>, based on dynamic programming, that tries to shuffle words around to minimise the amount of
		extra space at the end of each line.
	</li>
</ul>

<h2 id="wadler">Wadler's 'prettier printer', translated to Julia</h2>
{@render backToTop()}

<p>
	Please note that even though this is a word-for-word translation, it's not meant to have the same
	characteristics. In particular it's rather less efficient than the Haskell version because of the
	lack of laziness.
</p>

<p>
	The idea in Haskell is that if you have a <code>Union</code> doc, and you pick the first option,
	the second option never even needs to be materialised. This matters especially when you have
	nested <code>Union</code> docs. See the Lindig paper above.
</p>

<CodeExample anchorname={null} language={julia} filename="wadler.jl" code={wadler} />

<style>
	ul {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
</style>
