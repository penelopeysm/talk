<script lang="ts">
	import { base } from '$app/paths';
	import CodeExample from '$lib/CodeExample.svelte';
	import python from 'svelte-highlight/languages/python';

	import naive_py from './naive.py?raw';
	import fa_py from './fa.py?raw';
	import nfa_backtrack_py from './nfa_backtrack.py?raw';
	import nfa_linear_py from './nfa_linear.py?raw';

	const CODE_SNIPPETS = {
		naive_py: {
			anchor: 'naive-py',
			language: python,
			filename: 'naive.py',
			code: naive_py
		},
		fa_py: {
			anchor: 'fa-py',
			language: python,
			filename: 'fa.py',
			code: fa_py
		},
		nfa_backtrack_py: {
			anchor: 'nfa-backtrack-py',
			language: python,
			filename: 'nfa_backtrack.py',
			code: nfa_backtrack_py
		},
		nfa_linear_py: {
			anchor: 'nfa-linear-py',
			language: python,
			filename: 'nfa_linear.py',
			code: nfa_linear_py
		}
	};
</script>

<p><a href="{base}/">Back to list of talks</a></p>

<h1 id="toc">Table of contents - code snippets</h1>

{#snippet list_item(code)}
	<li><a href="#{code.anchor}"><code>{code.filename}</code></a></li>
{/snippet}
<ul>
	{@render list_item(CODE_SNIPPETS.naive_py)}
	{@render list_item(CODE_SNIPPETS.fa_py)}
	{@render list_item(CODE_SNIPPETS.nfa_backtrack_py)}
	{@render list_item(CODE_SNIPPETS.nfa_linear_py)}
</ul>

<br />
<hr />

<h1 id="links">Links</h1>
<ul>
	<li>
		<a href="https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/"
			>Cloudflare</a
		>
		and
		<a href="https://stackstatus.tumblr.com/post/147710624694/outage-postmortem-july-20-2016"
			>Stack Overflow</a
		> outages caused by rogue regexes
	</li>
	<li>
		<a href="https://swtch.com/~rsc/regexp/">Russ Cox's articles on regular expressions</a>: in
		particular, the
		<a href="https://swtch.com/~rsc/regexp/regexp1.html">first article in that list</a> forms part of
		the basis for this talk.
	</li>
	<li><a href="https://docs.rs/regex/latest/regex/">Documentation for Rust's regex crate</a></li>
	<li>
		<a href="https://dl.acm.org/doi/pdf/10.1145/3656431">Linear-time lookarounds in JavaScript</a>;
		there's currently
		<a href="https://github.com/rust-lang/regex/pull/1266">an open PR to the Rust regex crate</a>
	</li>
</ul>

<br />
<hr />

<h1>Code snippets</h1>

<CodeExample
	anchorname={CODE_SNIPPETS.naive_py.anchor}
	language={CODE_SNIPPETS.naive_py.language}
	filename={CODE_SNIPPETS.naive_py.filename}
	code={CODE_SNIPPETS.naive_py.code}
>
	<p>
		This is a very naive regex engine. It's way too hard-coded, and changing this to support
		different regular expressions is next-to-impossible, so we will look at ways of generalising
		this.
	</p>
</CodeExample>

<CodeExample
	anchorname={CODE_SNIPPETS.fa_py.anchor}
	language={CODE_SNIPPETS.fa_py.language}
	filename={CODE_SNIPPETS.fa_py.filename}
	code={CODE_SNIPPETS.fa_py.code}
>
	<p>
		This is a simplistic implementation of a finite automaton with states and transitions. In later
		sections we'll expand on this.
	</p>
</CodeExample>

<CodeExample
	anchorname={CODE_SNIPPETS.nfa_backtrack_py.anchor}
	language={CODE_SNIPPETS.nfa_backtrack_py.language}
	filename={CODE_SNIPPETS.nfa_backtrack_py.filename}
	code={CODE_SNIPPETS.nfa_backtrack_py.code}
>
	<p>
		This is a <i>backtracking</i> non-deterministic finite automaton (NFA). When it reaches a branch
		where multiple possible forward transitions are possible, it tries the first one. If that fails to
		parse it tries the next one, and so on. If all paths fail, it fails.
	</p>
	<p>
		This is accompanied by an implementation of various different regexes, to show that this engine
		is indeed general.
	</p>
</CodeExample>

<CodeExample
	anchorname={CODE_SNIPPETS.nfa_linear_py.anchor}
	language={CODE_SNIPPETS.nfa_linear_py.language}
	filename={CODE_SNIPPETS.nfa_linear_py.filename}
	code={CODE_SNIPPETS.nfa_linear_py.code}
>
	<p>
		This is an implementation of an NFA that doesn't backtrack. Instead of trying each branch in
		turn, we maintain a <i>set</i> of possible states, and progress through the NFA one character at
		a time. When we have exhausted the input, we check if any of the states are the <code>END</code>
		state. If so, then we have a match; if not, then we don't.
	</p>
</CodeExample>

<style>
	ul {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
</style>
