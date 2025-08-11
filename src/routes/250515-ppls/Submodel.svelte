<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import julia from 'svelte-highlight/languages/julia';

	const submodel_jl = `
using Turing

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function priors()
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    return (mu=mu, tau=tau)
end

@model function eight_schools_centered(J, y, sigma)
    p ~ to_submodel(priors())
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(p.mu, p.tau)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

@model function eight_schools_noncentered(J, y, sigma)
    p ~ to_submodel(priors())
    theta_trans = Vector{Float64}(undef, J)
    for i in 1:J
        theta_trans[i] ~ Normal(0, 1)
        theta = theta_trans[i] * p.tau + p.mu
        y[i] ~ Normal(theta, sigma[i])
    end
end
    `;
</script>

<CodeExample anchorname={null} language={julia} filename="submodel.jl" code={submodel_jl} />
