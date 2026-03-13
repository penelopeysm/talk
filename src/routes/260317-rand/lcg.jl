struct LCG
    a::Int
    c::Int
    m::Int
end
function next(rng::LCG, x::Int)
    return ((rng.a * x) + rng.c) % rng.m
end

rng = LCG(3, 5, 2^10)
seed = 3

for i in 1:20
    seed = next(rng, seed)
    println(seed % 8)
end
