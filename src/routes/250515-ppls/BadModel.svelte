<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import julia from 'svelte-highlight/languages/julia';

	const badmodel = `
using Turing

@model function f(x)
    y = x + 1
    y ~ Normal()
end

sample(f(1.0), NUTS(), 1000)
`;

</script>

<CodeExample
	anchorname={null}
	language={julia}
	filename="badmodel.py"
	code={badmodel}
/>
