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
