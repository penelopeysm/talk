from statistics import mean

class LCG:
    def __init__(self, a: int, c: int, m: int):
        self.a = a
        self.c = c
        self.m = m

    def next(self, x: int):
        return ((self.a * x) + self.c) % self.m


def generate_ints(N: int, seed: int):
    lcg = LCG(43, 1, 1024)
    x = seed
    samples = []
    for _ in range(N):
        x = lcg.next(x)
        samples.append(x)
    return samples

T = True
F = False

def to_bools(n: int):
    bools = []
    while n > 0:
        if (n % 2 == 1):
            bools.append(T)
        else:
            bools.append(F)
        n = n // 2
    return list(reversed(bools))

def from_bools(bs: list[bool]):
    exp = 0
    sum = 0
    for b in reversed(bs):
        if b:
            sum += 2**exp
        exp += 1
    return sum

class Float8:
    def __init__(self, sign: bool,
                 exponent: list[bool],
                 mantissa: list[bool]):
        assert len(exponent) == 4
        assert len(mantissa) == 3
        self.sign = sign
        self.exponent = exponent
        self.mantissa = mantissa

    def to_decimal(self):
        if not any(self.exponent):
            return 0.0

        sign = -1 if self.sign else +1
        exponent = 2 ** (from_bools(self.exponent) - 8)
        mantissa = 1 + (from_bools(self.mantissa) / 8)

        return sign * exponent * mantissa

    def __repr__(self):
        s = ""
        def _short(bool):
            return 'T' if bool else 'F'
        s += _short(self.sign)
        s += " " 
        s += "".join(map(_short, self.exponent))
        s += " " 
        s += "".join(map(_short, self.mantissa))
        s += " = "
        s += f"{self.to_decimal()}"
        return s

from itertools import product
for mantissa in product([F, T], repeat=3):
    print(Float8(F, [T, F, F, F], mantissa).to_decimal())

def generate_floats(N: int, seed: int):
    lcg = LCG(3, 1, 8)
    x = seed
    floats = []
    for _ in range(N):
        x = lcg.next(x)
        mantissa = to_bools(x)
        mantissa = ([F] * (3 - len(mantissa))) + mantissa
        floats.append(Float8(F, [T, F, F, F], mantissa))
    return floats

fs = generate_floats(20, 3)
for f in fs:
    print(f.to_decimal())
