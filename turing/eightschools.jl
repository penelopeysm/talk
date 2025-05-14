# Stan code: https://github.com/stan-dev/posteriordb/blob/master/posterior_database/models/stan/eight_schools_centered.stan

using PosteriorDB
pdb = PosteriorDB.database()
post = PosteriorDB.posterior(pdb, "eight_schools-eight_schools_centered")
data = PosteriorDB.dataset(post)
data_dict = PosteriorDB.load(data)

using Turing
@model function eight_schools_centered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(mu, tau)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

model_c = eight_schools_centered(data_dict["J"], data_dict["y"], data_dict["sigma"])

turing_chn = sample(model_c, NUTS(), MCMCThreads(), 1000, 10; num_warmup=10_000, thinning=10,)

ref = PosteriorDB.reference_posterior(post)
ref_post = PosteriorDB.load(ref)
param_names = sort(collect(keys(ref_post[1])))
n_iterations = length(ref_post[1][param_names[1]])
n_chains = length(ref_post)
vals = zeros(Float64, n_iterations, length(param_names), n_chains);
for (n_chain, dict) in enumerate(ref_post)
    for (n_param, p) in enumerate(param_names)
        vals[:, n_param, n_chain] = dict[p]
    end
end
pdb_chn = Chains(vals, param_names)


using Turing
@model function eight_schools_noncentered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta_trans = Vector{Float64}(undef, J)
    for i in 1:J
        theta_trans[i] ~ Normal(0, 1)
        theta = theta_trans[i] * tau + mu;
        y[i] ~ Normal(theta, sigma[i])
    end
end
model_nc = eight_schools_noncentered(data_dict["J"], data_dict["y"], data_dict["sigma"])
turing_chn = sample(model_nc, NUTS(), 1000; num_warmup=10_000, thinning=10,)

using MicroCanonicalHMC
spl = externalsampler(MCHMC(10000, 0.001))

turing_chn = sample(model_c, NUTS(), 1000; num_warmup=10_000, thinning=10,)

turing_chn = sample(model_c, spl, 1000; num_warmup=10_000, thinning=10,)


@model function eight_schools_priors()
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    return (mu=mu, tau=tau)
end

@model function eight_schools_centered(J, y, sigma)
    priors ~ to_submodel(eight_schools_priors())
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(priors.mu, priors.tau)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

@model function eight_schools_noncentered(J, y, sigma)
    priors ~ to_submodel(eight_schools_priors())
    theta_trans = Vector{Float64}(undef, J)
    for i in 1:J
        theta_trans[i] ~ Normal(0, 1)
        theta = theta_trans[i] * priors.tau + priors.mu;
        y[i] ~ Normal(theta, sigma[i])
    end
end

model_esc = eight_schools_centered(J, y, sigma)
model_esnc = eight_schools_noncentered(J, y, sigma)

chain_esc = sample(model_esc, NUTS(), 1000; num_warmup=10_000, thinning=10,)
@model function eight_schools_priors()
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    return (mu=mu, tau=tau)
end

@model function eight_schools_centered(J, y, sigma)
    priors ~ to_submodel(eight_schools_priors())
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(priors.mu, priors.tau)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

@model function eight_schools_noncentered(J, y, sigma)
    priors ~ to_submodel(eight_schools_priors())
    theta_trans = Vector{Float64}(undef, J)
    for i in 1:J
        theta_trans[i] ~ Normal(0, 1)
        theta = theta_trans[i] * priors.tau + priors.mu;
        y[i] ~ Normal(theta, sigma[i])
    end
end

model_esc = eight_schools_centered(J, y, sigma)
model_esnc = eight_schools_noncentered(J, y, sigma)

chain_esc = sample(model_esc, NUTS(), 1000; num_warmup=10_000, thinning=10,)
chain_esnc = sample(model_esnc, NUTS(), 1000; num_warmup=10_000, thinning=10,)
chain_esnc = sample(model_esnc, NUTS(), 1000; num_warmup=10_000, thinning=10,)
