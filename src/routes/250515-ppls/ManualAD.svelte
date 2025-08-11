<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import julia from 'svelte-highlight/languages/julia';

	const manualAD = `
using Turing
import DynamicPPL, LogDensityProblems

@model f() = x ~ Normal()

function LogDensityProblems.logdensity_and_gradient(
    ldf::DynamicPPL.LogDensityFunction{<:DynamicPPL.Model{typeof(f)}},
    x::AbstractVector
)
    # This function must return the log probability density as the first
    # argument and the gradient as the second argument.
    # You could manually calculate the first argument too, but here we 
    # just defer to the existing logdensity function.
    return LogDensityProblems.logdensity(ldf, x), [-x[1]]
end

sample(f(), NUTS(), 1000)
`;
</script>

<CodeExample anchorname={null} language={julia} filename="manual_ad.jl" code={manualAD} />
