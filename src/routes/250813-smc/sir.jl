using Distributions

"""
Implementation of sampling-importance-resampling (Bishop, section 11.1.5).

We want to sample from a distribution `p`, for which we can only evaluate
the logpdf `logpdf_p` (up to a constant factor). To do so, we: 

1. sample from a known distribution `q`
2. assign each sample the weight `w(x) = p(x) / q(x)`
3. resample from the weighted samples

For generalisability purposes we pass the various `logpdf` and `rand`
functions as arguments rather than passing the distribution `q` directly.
"""
function sir(
    logpdf_p::Function,
    rand_q::Function,
    logpdf_q::Function,
    n_samples::Int,
)
    # sample from q
    q_samples = [rand_q() for _ in 1:n_samples]
    # calculate the importance weights
    # ('adjusting' for difference between p and q)
    logpdfs_q = [logpdf_q(x) for x in q_samples]
    logpdfs_p = [logpdf_p(x) for x in q_samples]
    log_weights = logpdfs_p .- logpdfs_q
    weights = exp.(log_weights)
    # normalise the weights
    weights ./= sum(weights)
    # then resample from the weighted samples
    p_samples_idx = rand(Categorical(weights), length(q_samples))
    p_samples = q_samples[p_samples_idx]
    return p_samples
end

# We should check that it works. Let's say p is normal
logpdf_p = x -> -x^2
# And q is a (broad enough) uniform
q = Uniform(-10, 10)
rand_q = () -> rand(q)
logpdf_q = x -> logpdf(q, x)
normally_distributed = sir(logpdf_p, rand_q, logpdf_q, 1000)

using Plots
histogram(normally_distributed, bins=30)
