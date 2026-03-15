<script lang="ts">
	type Talk = {
		date: string;
		title: string;
		link: string;
		occasion?: string;
		hasRecording: boolean;
	};
	const TALKS: Talk[] = [
		{
			date: '17 March 2026',
			title:
				'What Really Goes On Inside Random Number Generation (and how to use that to win at Pokémon)',
			link: '260317-rand',
			occasion: 'REG tech talk',
			hasRecording: false
		},
		{
			date: '2 October 2025',
			title: 'Turing.jl poster (with Markus Hauru)',
			link: 'JuliaCon2025_Turing_Poster.pdf',
			occasion: 'JuliaCon Local Paris, 2025',
			hasRecording: false
		},
		{
			date: '1 October 2025',
			title: "Why we're NUTS about Turing.jl (and what you can infer from it)",
			link: '251001_nuts_turing.pdf',
			occasion: 'Fundamental Research showcase event, The Alan Turing Institute',
			hasRecording: false
		},
		{
			date: '13 August 2025',
			title: 'Sequential Monte Carlo from scratch',
			link: '250813-smc',
			occasion: 'PPL study group, The Alan Turing Institute',
			hasRecording: false
		},
		{
			date: '8 July 2025',
			title:
				'^What Really Goes On Inside A Regular Expression (and what that has to do with Alan Turing)?$',
			link: '250708-regex',
			occasion: 'REG tech talk',
			hasRecording: true
		},
		{
			date: '15 May 2025',
			title: 'Why Turing.jl? A limited comparison of probabilistic programming frameworks',
			link: '250515-ppls',
			occasion: 'Learn Bayes seminar series',
			hasRecording: false
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

Talks with video recordings are marked with a 📹.

<ul>
	{#each TALKS as talk (talk.link)}
		<li>
			{#if talk.hasRecording}
				📹
			{/if}
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
		gap: 20px;
	}
</style>
