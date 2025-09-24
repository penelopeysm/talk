<script lang="ts">
	type Talk = {
		date: string;
		title: string;
		link: string;
		occasion?: string;
	};
	const TALKS: Talk[] = [
		{
			date: '1 October 2025',
			title: 'Why we\'re NUTS about Turing.jl (and what you can infer from it)',
			link: '/251001_nuts_turing.pdf',
			occasion: 'Fundamental Research showcase event, The Alan Turing Institute'
		},
		{
			date: '13 August 2025',
			title: 'Sequential Monte Carlo from scratch',
			link: '250813-smc',
			occasion: 'PPL study group, The Alan Turing Institute'
		},
		{
			date: '8 July 2025',
			title:
				'^What Really Goes On Inside A Regular Expression (and what that has to do with Alan Turing)?$',
			link: '250708-regex',
			occasion: 'REG tech talk'
		},
		{
			date: '15 May 2025',
			title: 'Why Turing.jl? A limited comparison of probabilistic programming frameworks',
			link: '250515-ppls',
			occasion: 'Learn Bayes seminar series'
		}
	];
</script>

<h1 id="toc">Supplementary material for (some of) my talks</h1>

<p>
	This page hosts code blocks and links to other resources that I mention in my talks. Sometimes
	they have more elaboration, sometimes not.
</p>

<p>
	If anything is unclear (or, *gasp*, incorrect!), please let me know by opening an issue <a
		href="https://github.com/penelopeysm/talk"
		target="_blank">on GitHub</a
	>.
</p>

<ul>
	{#each TALKS as talk (talk.link)}
		<li>
			<a href={talk.link}>
				{talk.date}: <i>{talk.title}</i>
				{#if talk.occasion}
					({talk.occasion}){/if}
			</a>
		</li>
	{/each}
</ul>

<style>
	ul {
		margin-top: 40px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
</style>
