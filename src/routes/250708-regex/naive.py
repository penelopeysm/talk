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
