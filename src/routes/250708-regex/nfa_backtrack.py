from __future__ import annotations

nchecks = 0

class MatchSuccess(Exception):
    pass

class MatchFailure(Exception):
    pass

STATES = dict()

def parse(state: str, input: str):
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
