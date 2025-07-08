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
