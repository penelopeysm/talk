<script lang="ts">
	import { base } from '$app/paths';
	import CodeExample from '$lib/CodeExample.svelte';

	import atomOneLight from 'svelte-highlight/styles/atom-one-light';
	import diff from 'svelte-highlight/languages/diff';
	import python from 'svelte-highlight/languages/python';

	const cpython_patch = `
diff --git a/Lib/re/_compiler.py b/Lib/re/_compiler.py
index 1b1aaa7714b..b0d5578eae1 100644
--- a/Lib/re/_compiler.py
+++ b/Lib/re/_compiler.py
@@ -36,7 +36,10 @@ def _combine_flags(flags, add_flags, del_flags,
 
 def _compile(code, pattern, flags):
     # internal: compile a (sub)pattern
-    emit = code.append
+    print("calling _compile!")
+    def emit(x):
+        print("emit:", x)
+        code.append(x)
     _len = len
     LITERAL_CODES = _LITERAL_CODES
     REPEATING_CODES = _REPEATING_CODES
`;

	const simple_py = String.raw`
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

	const dfa_py = String.raw`
from __future__ import annotations

class MatchSuccess(Exception):
    pass

class MatchFailure(Exception):
    pass

class SuccessState:
    def parse(self, input):
        raise MatchSuccess()

class EndState:
    def parse(self, input):
        if input == "":
            return SuccessState().parse(input)
        else:
            raise MatchFailure()

class State:
    transition: tuple[str, Union[EndState,State]]

    def __eq__(self, other):
        return self.transition == other.transition

    def __init__(self, transition):
        self.transition = transition

    def parse(self, input: str) -> tuple[State, str]:
        expected, next_state = self.transition

        if input.startswith(expected):
            remaining_input = input.removeprefix(expected)
            return next_state.parse(remaining_input)
        else:
            raise MatchFailure()

def match_regex(start_state: State, input_string: str) -> bool:
    try:
        start_state.parse(input_string)
    except MatchSuccess:
        return True
    except MatchFailure:
        return False

R = State(("c", EndState()))
Q = State(("b", R))
P = State(("a", Q))

print("----- \"abc\" ------")
for s in ["abc", "abcd", "ac"]:
    print(s, match_regex(P, s))
`;

	const nfa_backtrack_py = String.raw`
from __future__ import annotations

nchecks = 0

class MatchSuccess(Exception):
    pass

class MatchFailure(Exception):
    pass

STATES = dict()

def parse(state: str, input: str) -> tuple[str, str]:
    global nchecks

    if state == "SUCCESS":
        raise MatchSuccess()

    if state == "END":
        nchecks += 1
        if input == "":
            parse("SUCCESS", "")
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

	const nfa_thompson_py = String.raw`
from __future__ import annotations

nchecks = 0

class MatchSuccess(Exception):
    pass

STATES = dict()

def get_all_empty_transitions(state: str) -> set[str]:
    if state == "SUCCESS" or state == "END":
        return set()
    else:
        next_states = set(next_state for (expected, next_state) in STATES[state] if expected == "")
        empty_transitions = set()
        for t in next_states:
            empty_transitions.update(get_all_empty_transitions(t))
        empty_transitions.add(state)
        return empty_transitions

def progress(current_states: set[str], input: str) -> set[str]:
    global nchecks

    possible_outcomes = set()

    for state in current_states:
        if state == "SUCCESS":
            possible_outcomes.add("SUCCESS")
            continue

        if state == "END":
            nchecks += 1
            if input == "":
                possible_outcomes.add("SUCCESS")
            continue

        if state not in STATES:
            raise RuntimeError(f"Oops! You reached an invalid state: {state}.")

        this_state_outcomes = set()
        for expected, next_state in STATES[state]:
            nchecks += 1
            if input.startswith(expected):
                this_state_outcomes.add(next_state)
                this_state_outcomes.update(get_all_empty_transitions(next_state))

        possible_outcomes.update(this_state_outcomes)

    if any(state == "SUCCESS" for state in possible_outcomes):
        raise MatchSuccess()

    return possible_outcomes

def match_regex(start_state: str, input: str) -> bool:
    global nchecks
    nchecks = 0
    states = get_all_empty_transitions(start_state)
    try:
        while len(states) > 0:
            states = progress(states, input)
            input = input[1:]
        return False
    except MatchSuccess:
        return True

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

	const CODE_SNIPPETS = {
		cpython_patch: {
			anchor: 'cpython-patch',
			language: diff,
			filename: 'cpython_regex.patch',
			code: cpython_patch
		},
		simple_py: {
			anchor: 'simple-py',
			language: python,
			filename: 'simple.py',
			code: simple_py
		},
		dfa_py: {
			anchor: 'dfa-py',
			language: python,
			filename: 'dfa.py',
			code: dfa_py
		},
		nfa_backtrack_py: {
			anchor: 'nfa-backtrack-py',
			language: python,
			filename: 'nfa_backtrack.py',
			code: nfa_backtrack_py
		},
		nfa_thompson_py: {
			anchor: 'nfa-thompson-py',
			language: python,
			filename: 'nfa_thompson.py',
			code: nfa_thompson_py
		}
	};
</script>

<svelte:head>
	{@html atomOneLight}
</svelte:head>

<p><a href="{base}/">Back to list of talks</a></p>

<h1 id="toc">Table of contents - code snippets</h1>

{#snippet list_item(code)}
	<li><a href="#{code.anchor}">{code.filename}</a></li>
{/snippet}
<ul>
	{@render list_item(CODE_SNIPPETS.simple_py)}
	{@render list_item(CODE_SNIPPETS.dfa_py)}
	{@render list_item(CODE_SNIPPETS.nfa_backtrack_py)}
	{@render list_item(CODE_SNIPPETS.cpython_patch)}
	{@render list_item(CODE_SNIPPETS.nfa_thompson_py)}
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
		<a href="https://github.com/python/cpython/blob/3.13/Lib/re/_compiler.py"
			>CPython's regex compiler</a
		>
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

{#snippet simple_desc()}
	<p>
		This is a very simple regex engine. It's way too hard-coded, and changing this to support
		different regular expressions is next-to-impossible, so we will look at ways of generalising
		this.
	</p>
{/snippet}
<CodeExample
	anchorname={CODE_SNIPPETS.simple_py.anchor}
	language={CODE_SNIPPETS.simple_py.language}
	filename={CODE_SNIPPETS.simple_py.filename}
	code={CODE_SNIPPETS.simple_py.code}
	description={simple_desc}
/>

{#snippet dfa_desc()}
	<p>
		This is a simplistic implementation of a deterministic finite automaton (DFA), i.e., one where
		the there is no ambiguity as to which state to transition to next. (In fact, deterministic
		finite automata can have multiple onwards transitions for a single state: it just means that it
		should not be possible for more than one of them to be applicable.)
	</p>
{/snippet}
<CodeExample
	anchorname={CODE_SNIPPETS.dfa_py.anchor}
	language={CODE_SNIPPETS.dfa_py.language}
	filename={CODE_SNIPPETS.dfa_py.filename}
	code={CODE_SNIPPETS.dfa_py.code}
	description={dfa_desc}
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

{#snippet cpython_patch_desc()}
	<p>
		Extra printing statements which I added to CPython to illustrate the regex compilation process.
		This diff was applied to the <code>3.13</code>
		branch of <a href="https://github.com/python/cpython">CPython</a>.
	</p>
	<p>
		To replicate this, you can clone the repo, run <code
			>git checkout da2c4ef7eb4; git apply cpython_regex.patch</code
		>, and then compile it with <code>./configure; make</code>. That will create a patched Python
		interpreter which you can run with <code>./python.exe</code>.
	</p>
	<p>
		Then, running <code>import re; re.compile("a?a?a?aaa")</code> will print output that is not too
		dissimilar to the <code>STATES</code> dictionary in the previous snippet.
	</p>
{/snippet}
<CodeExample
	anchorname={CODE_SNIPPETS.cpython_patch.anchor}
	language={CODE_SNIPPETS.cpython_patch.language}
	filename={CODE_SNIPPETS.cpython_patch.filename}
	code={CODE_SNIPPETS.cpython_patch.code}
	description={cpython_patch_desc}
/>

{#snippet nfa_thompson_desc()}
	<p>
		This is an implementation of an NFA that doesn't backtrack. Instead of trying each branch in
		turn, we maintain a <i>set</i> of possible states. Success is reached when the set of possible
		states contains the <code>SUCCESS</code> state; failure is reached when the set of possible states
		is empty.
	</p>
{/snippet}
<CodeExample
	anchorname={CODE_SNIPPETS.nfa_thompson_py.anchor}
	language={CODE_SNIPPETS.nfa_thompson_py.language}
	filename={CODE_SNIPPETS.nfa_thompson_py.filename}
	code={CODE_SNIPPETS.nfa_thompson_py.code}
	description={nfa_thompson_desc}
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
