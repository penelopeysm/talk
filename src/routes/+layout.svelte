<script lang="ts">
    import { base } from '$app/paths';
    let { children } = $props();
</script>

<svelte:head>
    <title>Talks</title>
    <link rel="stylesheet" href="{base}/main.css" />
</svelte:head>

{@render children()}
