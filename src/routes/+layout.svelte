<script lang="ts">
	import { base } from '$app/paths';
	let { children } = $props();
</script>

<svelte:head>
	<title>Talks</title>
	<link rel="stylesheet" href="{base}/main.css" />
</svelte:head>

{@render children()}

<style>
	:global(html) {
		background-color: var(--background);
		color: var(--text);
	}
	:global(a:link) {
		color: var(--a-link);
	}
	:global(a:visited) {
		color: var(--a-visited);
	}
</style>
