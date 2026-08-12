<script lang="ts">
	import { base } from '$app/paths';
</script>

<p id="top"><a href="{base}/">Back to list of talks</a></p>
{#snippet backToTop()}
	<a href="#top">Back to top</a>
{/snippet}

Since this is quite a short talk, I don't actually have time to show code excerpts.
Sorry!

<h1 id="links">Links</h1>
<ul>
	<li>
		<a href="https://pysm.dev/posts/2026-04-30-turing/"
			>My original blog post, a longer version of this talk.</a
		>
	</li>
	<li>
		<a href="JuliaCon2026_Turing.pdf">Slides (PDF)</a>
	</li>
</ul>

You might also like <a href="JuliaCon2025_Turing_Poster.pdf">our poster from last year's JuliaCon Local in Paris</a> which lists a few more library features (although please note that the 'future plans' section might not actually happen any more).

<style>
	ul {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
</style>
