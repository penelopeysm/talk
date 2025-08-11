<script lang="ts">
	import CodeExample from '$lib/CodeExample.svelte';
	import python from 'svelte-highlight/languages/python';

	const eight_schools_centered_pymc = `
import pymc as pm
import numpy as np
import time

J = 8
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

def main():
    with pm.Model() as eight_schools_centered:
        mu = pm.Normal("mu", 0, 5)
        tau = pm.HalfCauchy("tau", 5)
        theta = pm.Normal("theta", mu, tau, shape=J)
        obs = pm.Normal("obs", theta, sigma, observed=y)

        start = time.time()
        trace = pm.sample(draws=10000, tune=10000, chains=1)
        end = time.time()

    print(f"took {end - start} seconds")

if __name__ == "__main__":
    main()`;

</script>

<CodeExample
	anchorname={null}
	language={python}
	filename="eight_schools_centered_pymc.py"
	code={eight_schools_centered_pymc}
/>
