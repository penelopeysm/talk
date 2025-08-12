<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import julia from 'svelte-highlight/languages/julia';

	const submodel_naive_jl = `
using Turing

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function eight_schools_noncentered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    # We have to change the name of this vector
    theta_trans = Vector{Float64}(undef, J)
    for i in 1:J
        # This was our original model:
        # theta[i] ~ Normal(mu, tau)
        # We changed it to:
        theta_trans[i] ~ Normal(0, 1)
        theta_i = theta_trans[i] * tau + mu
        # This likelihood term remains the same.
        y[i] ~ Normal(theta_i, sigma[i])
    end
end

model_esnc = eight_schools_noncentered(J, y, sigma)

chain = sample(model_esnc, NUTS(), 1000; num_warmup=10000, thinning=10)
ess(chain)
`;
</script>

<CodeExample
	anchorname={null}
	language={julia}
	filename="submodel_naive.jl"
	code={submodel_naive_jl}
/>
