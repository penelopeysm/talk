include("particle_filter.jl")

"""
Particle marginal Metropolis-Hastings (PMMH) algorithm.
"""
function pmmh(n_particles::Int, ys::Vector{Float64}, n_iters::Int)
    samples = Particle[]

    # First iteration: just run the particle filter
    particles, logevidence = run_particle_filter(n_particles, ys)
    # Pick a random particle (with uniform weights)
    this_iter = rand(particles)
    push!(samples, this_iter)

    for _ in 2:n_iters
        # Run the particle filter again
        new_particles, new_logevidence = run_particle_filter(n_particles, ys)

        # Calculate MH factor
        acceptance_ratio = exp(new_logevidence - logevidence)
        
        # Accept or reject the new sample
        if rand() < acceptance_ratio
            # Accept the new sample
            particles = new_particles
            logevidence = new_logevidence
            this_iter = rand(particles)
        else
            # Reject the new sample, keep the old one
            this_iter = deepcopy(samples[end])
        end

        # Add to samples
        push!(samples, this_iter)
    end
    return samples
end

println("------ Hand coded PMMH ------")
ys = [1.0, 2.0, 3.0, 4.0, 5.0]  # Example observations
chain = pmmh(20, ys, 1000)
for i in 1:length(ys)
    xi_samples = [p.xs[i] for p in chain]
    println("x[$i] = $(mean(xi_samples)) ± $(std(xi_samples))")
end

println("------ Turing NUTS ------")
using Turing
@model function ssm(ys)
    xs = Vector{Float64}(undef, length(ys))
    xs[1] ~ Normal()
    ys[1] ~ Normal(xs[1])
    for t in 2:length(ys)
        xs[t] ~ Normal(xs[t-1], 1.0)
        ys[t] ~ Normal(xs[t])
    end
end
chn = sample(ssm(ys), NUTS(), 1000)
describe(chn)
