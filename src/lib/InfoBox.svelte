<script lang="ts">
	import type { Snippet } from 'svelte';

	interface InfoBoxProps {
		title: string;
		isWarning?: boolean;
		children?: Snippet;
	}
	const { title, isWarning = false, children }: InfoBoxProps = $props();
</script>

<div id="box" class={isWarning ? 'warning' : 'info'}>
	<h3>{title}</h3>
	{#if children}
		{@render children()}
	{/if}
</div>

<style>
	div#box {
		margin: 20px;
		border-radius: 10px;
		padding: 10px 20px;
		font-size: 0.9em;
		color: var(--text);

		h3 {
			margin: 0;
		}

		:global(:last-child) {
			margin-bottom: 0;
		}
	}

	div.info {
		background-color: var(--info-box-background);
	}

	div.warning {
		background-color: var(--warning-box-background);
	}
</style>
