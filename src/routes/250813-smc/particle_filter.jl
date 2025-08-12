using Distributions

function sir_minimal(
    q_samples::AbstractVector,
    log_weights::AbstractVector,
)
    @assert length(q_samples) == length(log_weights)
    # sample from q
    weights = exp.(log_weights)
    # normalise the weights
    weights ./= sum(weights)
    # then resample from the weighted samples
    p_samples_idx = rand(Categorical(weights), length(q_samples))
    p_samples = q_samples[p_samples_idx]
    return [deepcopy(p) for p in p_samples]
end

# p.xs[i] == value of x[i] for this particle
struct Particle
    xs::Vector{Float64}
end

function initialise_particles(n_particles::Int)
    return [Particle([]) for _ in 1:n_particles]
end

function sample_new_x!(particles::Vector{Particle}, i::Int)
    # check that we're sampling the next x
    @assert all(p -> length(p.xs) == i - 1, particles)
    for p in particles
        # 'bootstrap filter' i.e. sample x from its prior distribution
        # then we don't need to include it in the weight
        x_i = if i == 1
            # x_1 ~ Normal(0, 1)
            rand(Normal())
        else
            # x_i ~ Normal(x_{i-1}, 1)
            rand(Normal(p.xs[i-1]))
        end
        push!(p.xs, x_i)
    end
end

function get_logweights(
    particles::Vector{Particle},
    i::Int,
    ys::Vector{Float64}
)
    # y_i ~ Normal(x_i, 1)
    # the weight is just the likelihood
    return map(p -> logpdf(Normal(p.xs[i]), ys[i]), particles)
end

function run_particle_filter(n_particles::Int, ys::Vector{Float64})
    # initialise particles
    particles = initialise_particles(n_particles)
    # number of observations
    n_obs = length(ys)
    # initialise log-evidence
    log_evidence = 0.0
    for i in 1:n_obs
        # sample new x_i for each particle
        sample_new_x!(particles, i)
        # calculate weights for y_i
        logweights = get_logweights(particles, i, ys)
        # calculate log-evidence
        log_evidence += log(mean(exp.(logweights)))
        # resample based on those weights
        particles = sir_minimal(particles, logweights)
    end
    return particles, log_evidence
end

pf, le = run_particle_filter(50, [1.0, 2.0, 3.0, 4.0, 5.0])
for particle in pf
    display(particle)
end
println("Log-evidence: ", le)
