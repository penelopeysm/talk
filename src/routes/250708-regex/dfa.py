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
