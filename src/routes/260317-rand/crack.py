class LCG:
    def __init__(self, a: int, c: int, m: int):
        self.a = a
        self.c = c
        self.m = m

    def next(self, x: int):
        return ((self.a * x) + self.c) % self.m


def crack(a, c, seed):
    # Fix m to be a prime. (Notice some of the parameters in the LCG table are
    # (2^N - 1) -- 'Mersenne primes'!)
    m = 101
    assert 0 <= a < m
    assert 0 <= c < m
    assert 0 <= seed < m
    lcg = LCG(a, c, m) 
    x1 = lcg.next(seed)
    x2 = lcg.next(x1)
    x3 = lcg.next(x2)
    x4 = lcg.next(x3)
    print(f"Give these to Penny: x1={x1}, x2={x2}, x3={x3}")
    print()
    print(f"Keep these a secret: a={a}, c={c}, x4={x4}")

# Replace these with your own choices
crack(2, 3, 5)
