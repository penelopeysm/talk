from statistics import mean

class LCG:
    def __init__(self, a: int, c: int, m: int):
        self.a = a
        self.c = c
        self.m = m

    def next(self, x: int):
        return ((self.a * x) + self.c) % self.m
    
lcg = LCG(43, 7, 1024)

def draw_int_samples(N, seed=5):
    x = seed
    samples = []
    for _ in range(N):
        samples.append(x)
        x = lcg.next(x)
    return samples

samples = draw_int_samples(500)
expected_mean = (1023 - 0) / 2
print(expected_mean)
print(mean(samples))

print("------")

def draw_float_samples(N, seed=5):
    return [i/1024 for i in draw_int_samples(N, seed=seed)]

samples = draw_float_samples(500)
expected_mean = 0.5
print(expected_mean)
print(mean(samples))

# ------------------------

def bools_to_int(bools: list[bool]):
    exponent = 0
    sum = 0
    for b in reversed(bools):
        if b:
            sum += 2**exponent
        exponent += 1
    return sum

T = True
F = False

bools_to_int([T, T, F])


import math

class Float8:
    def __init__(self, sign: bool, exponent: list[bool],
                 mantissa: list[bool]):
        assert len(exponent) == 3
        assert len(mantissa) == 4
        self.sign = sign
        self.exponent = exponent
        self.mantissa = mantissa

    def to_decimal(self):
        if not any(self.exponent):
            return 0.0
        if all(self.exponent):
            return math.inf

        sign = -1 if self.sign else +1
        exponent = bools_to_int(self.exponent) - 4
        mantissa = 1 + (bools_to_int(self.mantissa) / 16)
        return sign * (mantissa) * 2**exponent


print(Float8(F, [F, F, T], [F, F, F, F]).to_decimal())
