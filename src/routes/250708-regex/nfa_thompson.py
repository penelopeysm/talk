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
