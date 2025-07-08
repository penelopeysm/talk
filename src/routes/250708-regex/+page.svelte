<script lang="ts">
	import { base } from '$app/paths';
	import CodeExample from '$lib/CodeExample.svelte';

	import atomOneLight from 'svelte-highlight/styles/atom-one-light';
	import diff from 'svelte-highlight/languages/diff';
	import python from 'svelte-highlight/languages/python';

	const naive_py = String.raw`
"""
Verify if a string matches the regular expression 'abc'.
"""
def parse_abc(to_match: str):
    if to_match.startswith("a"):
        to_match = to_match.removeprefix("a")
        if to_match.startswith("b"):
            to_match = to_match.removeprefix("b")
            if to_match.startswith("c"):
                to_match = to_match.removeprefix("c")
                if to_match == "":
                    return True
    return False

print("----- \"abc\" ------")
for s in ["abc", "abcd", "ac"]:
    print(s, parse_abc(s))

"""
Verify if a string matches the regular expression 'ab?c'.
"""
def parse_abQc(to_match: str):
    if to_match.startswith("a"):
        to_match = to_match.removeprefix("a")

        if to_match.startswith("b"):
            to_match = to_match.removeprefix("b")
            if to_match.startswith("c"):
                to_match = to_match.removeprefix("c")
                if to_match == "":
                    return True
        else:
            if to_match.startswith("c"):
                to_match = to_match.removeprefix("c")
                if to_match == "":
                    return True
    return False

print("----- \"ab?c\" ------")
for s in ["abc", "abcd", "ac"]:
    print(s, parse_abQc(s))
`;

	const fa_py = String.raw`
class MatchSuccess(Exception):
    pass

class MatchFailure(Exception):
    pass

STATES = {}

def parse(state: str, input: str):
    if state == "END":
        if input == "":
            raise MatchSuccess()
        else:
            raise MatchFailure()

    if state not in STATES:
        raise RuntimeError(f"Oops! You reached an invalid state: {state}.")

    expected, next_state = STATES[state]
    if input.startswith(expected):
        remaining_input = input.removeprefix(expected)
        return parse(next_state, remaining_input)
    else:
        raise MatchFailure()

def match_regex(start_state: str, input_string: str) -> bool:
    try:
        parse(start_state, input_string)
    except MatchSuccess:
        return True
    except MatchFailure:
        return False

print("----- \"abc\" ------")
STATES["P"] = ("a", "Q")
STATES["Q"] = ("b", "R")
STATES["R"] = ("c", "END")
for s in ["abc", "abcd", "ac"]:
    print(s, match_regex("P", s))
`;

	const nfa_backtrack_py = String.raw`
class MatchSuccess(Exception):
    pass

class MatchFailure(Exception):
    pass

STATES = dict()
nchecks = 0

def parse(state: str, input: str):
    global nchecks

    if state == "END":
        nchecks += 1
        if input == "":
            raise MatchSuccess()
        else:
            raise MatchFailure()

    if state not in STATES:
        raise RuntimeError(f"Oops! You reached an invalid state: {state}.")

    possible_transitions = STATES[state]
    for expected, next_state in possible_transitions:
        try:
            nchecks += 1
            if input.startswith(expected):
                remaining_input = input.removeprefix(expected)
                return parse(next_state, remaining_input)
            else:
                raise MatchFailure()
        except MatchFailure:
            continue
    else:
        raise MatchFailure()

def match_regex(state: str, input: str) -> bool:
    global nchecks
    nchecks = 0
    try:
        parse(state, input)
    except MatchSuccess:
        return True
    except MatchFailure:
        return False

print("----- \"abc\" ------")
STATES["P"] = [("a", "Q")]
STATES["Q"] = [("b", "R")]
STATES["R"] = [("c", "END")]
for s in ["abc", "abcd", "ac"]:
    print(s, match_regex("P", s), "checks:", nchecks)

print("----- \"ab?c\" ------")
STATES["P2"] = [("a", "Q2")]
STATES["Q2"] = [("b", "R2"), ("", "R2")]
STATES["R2"] = [("c", "END")]
for s in ["abc", "abcd", "ac"]:
    print(s, match_regex("P2", s), "checks:", nchecks)

print("----- \"ab+c\" ------")
STATES["P3"] = [("a", "Q3")]
STATES["Q3"] = [("b", "R3")]
STATES["R3"] = [("c", "END"), ("", "Q3")]
for s in ["ac", "abc", "abbbbbbc"]:
    print(s, match_regex("P3", s), "checks:", nchecks)

print("----- \"a?a?a?aaa\" (greedy) ------")
STATES["Aq1"] = [("a", "Aq2"), ("", "Aq2")]
STATES["Aq2"] = [("a", "Aq3"), ("", "Aq3")]
STATES["Aq3"] = [("a", "Aq4"), ("", "Aq4")]
STATES["Aq4"] = [("a", "Aq5")]
STATES["Aq5"] = [("a", "Aq6")]
STATES["Aq6"] = [("a", "END")]
for s in ["aaa", "aaaa", "aaaaa", "aaaaaa"]:
    print(s, match_regex("Aq1", s), "checks:", nchecks)

print("----- \"a?a?a?aaa\" (nongreedy) ------")
STATES["Aqq1"] = [("", "Aqq2"), ("a", "Aqq2")]
STATES["Aqq2"] = [("", "Aqq3"), ("a", "Aqq3")]
STATES["Aqq3"] = [("", "Aqq4"), ("a", "Aqq4")]
STATES["Aqq4"] = [("a", "Aqq5")]
STATES["Aqq5"] = [("a", "Aqq6")]
STATES["Aqq6"] = [("a", "END")]
for s in ["aaa", "aaaa", "aaaaa", "aaaaaa"]:
    print(s, match_regex("Aqq1", s), "checks:", nchecks)

print("----- \"(a?)_n(a)_n\" ------")
for n in range(1, 20):
    for i in range(n):
        STATES[f"Aq{n}_{i}"] = [("a", f"Aq{n}_{i+1}"), ("", f"Aq{n}_{i+1}")]
        if i == n - 1:
            STATES[f"Aq{n}_{n+i}"] = [("a", "END")]
        else:
            STATES[f"Aq{n}_{n+i}"] = [("a", f"Aq{n}_{n+i+1}")]
    test_string = "a" * n
    print(f"n={n}:", test_string, match_regex(f"Aq{n}_0", test_string), "checks:", nchecks)
`;

	const nfa_linear_py = String.raw`
class MatchSuccess(Exception):
    pass

class MatchFailure(Exception):
    pass

STATES = {}
nchecks = 0

def get_epsilon_closure(state: str):
    global nchecks
    if state == "END":
        return set(["END"])
    ts = set([state])
    for expected, next_state in STATES[state]:
        nchecks += 1
        if expected == "":
            ts.add(next_state)
            ts.update(get_epsilon_closure(next_state))
    return ts

def progress_state(in_states: set[str], input: str):
    global nchecks
    
    nchecks += 1
    if input == "":
        if any(s == "END" for s in in_states):
            raise MatchSuccess()
        else:
            raise MatchFailure()

    out_states = set()

    for state in in_states:
        if state == "END":
            out_states.add("END")
            continue

        if state not in STATES:
            raise ValueError(f"oops: {state}")

        for expected, next_state in STATES[state]:
            nchecks += 1
            if input[0] == expected:
                out_states.add(next_state)
                out_states.update(get_epsilon_closure(next_state))

    return progress_state(out_states, input[1:])

def match(start_state: str, input: str):
    global nchecks
    nchecks = 0
    try:
        progress_state(get_epsilon_closure(start_state), input)
    except MatchSuccess:
        print(f"{input} => True with {nchecks} checks")
        return True
    except MatchFailure:
        print(f"{input} => False with {nchecks} checks")
        return False

def create_states(n):
    base = f"A{n}"
    states = {}
    for i in range(n):
        states[f"{base}_{i}"] = [
            ("a", f"{base}_{i+1}"),
            ("", f"{base}_{i+1}"),
        ]
        if i == n - 1:
            states[f"{base}_{i+n}"] = [
                ("a", "END"),
            ]
        else:
            states[f"{base}_{i+n}"] = [
                ("a", f"{base}_{i+n+1}"),
            ]
    return states

for i in range(1, 41):
    STATES = STATES | create_states(i)

for i in range(1, 41):
    start_state = f"A{i}_0"
    for n in range(i, 2*i + 1):
        if not match(start_state, "a" * n):
            raise ValueError(f"Failed to match {i}, {n}")
`;

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

<svelte:head>
	{@html atomOneLight}
</svelte:head>

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

{#snippet naive_desc()}
	<p>
		This is a very naive regex engine. It's way too hard-coded, and changing this to support
		different regular expressions is next-to-impossible, so we will look at ways of generalising
		this.
	</p>
{/snippet}
<CodeExample
	anchorname={CODE_SNIPPETS.naive_py.anchor}
	language={CODE_SNIPPETS.naive_py.language}
	filename={CODE_SNIPPETS.naive_py.filename}
	code={CODE_SNIPPETS.naive_py.code}
	description={naive_desc}
/>

{#snippet fa_desc()}
	<p>
		This is a simplistic implementation of a finite automaton with states and transitions. In later
		sections we'll expand on this.
	</p>
{/snippet}
<CodeExample
	anchorname={CODE_SNIPPETS.fa_py.anchor}
	language={CODE_SNIPPETS.fa_py.language}
	filename={CODE_SNIPPETS.fa_py.filename}
	code={CODE_SNIPPETS.fa_py.code}
	description={fa_desc}
/>

{#snippet nfa_backtrack_desc()}
	<p>
		This is a <i>backtracking</i> non-deterministic finite automaton (NFA). When it reaches a branch
		where multiple possible forward transitions are possible, it tries the first one. If that fails to
		parse it tries the next one, and so on. If all paths fail, it fails.
	</p>
	<p>
		This is accompanied by an implementation of various different regexes, to show that this engine
		is indeed general.
	</p>
{/snippet}
<CodeExample
	anchorname={CODE_SNIPPETS.nfa_backtrack_py.anchor}
	language={CODE_SNIPPETS.nfa_backtrack_py.language}
	filename={CODE_SNIPPETS.nfa_backtrack_py.filename}
	code={CODE_SNIPPETS.nfa_backtrack_py.code}
	description={nfa_backtrack_desc}
/>

{#snippet nfa_linear_desc()}
	<p>
		This is an implementation of an NFA that doesn't backtrack. Instead of trying each branch in
		turn, we maintain a <i>set</i> of possible states, and progress through the NFA one character at
		a time. When we have exhausted the input, we check if any of the states are the <code>END</code>
		state. If so, then we have a match; if not, then we don't.
	</p>
{/snippet}
<CodeExample
	anchorname={CODE_SNIPPETS.nfa_linear_py.anchor}
	language={CODE_SNIPPETS.nfa_linear_py.language}
	filename={CODE_SNIPPETS.nfa_linear_py.filename}
	code={CODE_SNIPPETS.nfa_linear_py.code}
	description={nfa_linear_desc}
/>

<style>
	code {
		font-size: 0.9em;
		background-color: #f0f0f0;
	}
	ul {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
</style>
