<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import julia from 'svelte-highlight/languages/julia';

	const ad_jl = `
using Turing, ADTypes
import Enzyme: set_runtime_activity, Reverse

J = 8
y = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18]

@model function eight_schools_noncentered(J, y, sigma)
    mu ~ Normal(0, 5)
    tau ~ truncated(Cauchy(0, 5); lower=0)
    theta_trans = Vector{Float64}(undef, J)
    for i in 1:J
        theta_trans[i] ~ Normal(0, 1)
        theta = theta_trans[i] * tau + mu
        y[i] ~ Normal(theta, sigma[i])
    end
end

model = eight_schools_noncentered(J, y, sigma)

# ForwardDiff.jl is the default AD backend in Turing.jl.
# If you don't specify an AD backend, it will do the same as this.
forwarddiff = AutoForwardDiff()
chain = sample(model, NUTS(; adtype=forwarddiff), 2000)

# You can switch AD backends by passing a different adtype argument.
enzyme_reverse = AutoEnzyme(; mode=set_runtime_activity(Reverse, true))
chain = sample(model, NUTS(; adtype=enzyme_reverse), 2000)
    `;
</script>

<CodeExample anchorname={null} language={julia} filename="ad.jl" code={ad_jl} />
