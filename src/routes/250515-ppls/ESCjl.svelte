<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import julia from 'svelte-highlight/languages/julia';

	const eight_schools_centered_julia = `
using Turing

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function eight_schools_centered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta = Vector{Float64}(undef, J)
    for i in 1:J
        theta[i] ~ Normal(mu, tau)
        y[i] ~ Normal(theta[i], sigma[i])
    end
end

model_esc = eight_schools_centered(J, y, sigma)

chain = sample(model_esc, NUTS(), 1000; num_warmup=10000, thinning=10)
    `;
</script>

<CodeExample
	anchorname={null}
	language={julia}
	filename="eight_schools_centered.jl"
	code={eight_schools_centered_julia}
/>
