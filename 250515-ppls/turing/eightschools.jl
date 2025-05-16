using Turing

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function hyperparameters()
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    return (mu=mu, tau=tau)
end

prior_model = hyperparameters()
sample(prior_model, NUTS(), 1000)

@model function eight_schools_centered(J, y, sigma)
    p ~ to_submodel(hyperparameters())
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(p.mu, p.tau)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

@model function eight_schools_noncentered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(0, 1)
        theta2 = theta[i] * tau + mu
        @show theta2
        y[i] ~ Normal(theta2, sigma[i])
    end
end

m = eight_schools_centered(J, y, sigma)

import Enzyme: set_runtime_activity, Reverse
enzyme_reverse = AutoEnzyme(; mode=set_runtime_activity(Reverse, true))

chain = sample(m2, NUTS(), 5000)

m2 = eight_schools_noncentered(J, y, sigma)

chain = sample(m2, NUTS(; adtype=enzyme_reverse), 5000)

@model function g(x)
    x ~ Normal(1)
end


@model function f(x)
    y = x
    y ~ Normal(1)
end
