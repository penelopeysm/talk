<script lang="ts">
	import type { Snippet } from 'svelte';
	import Highlight from 'svelte-highlight';
	import type { LanguageType } from 'svelte-highlight/languages';
	interface CodeExampleProps {
		anchorname: string | null;
		language: LanguageType<string>;
		filename: string;
		code: string;
		children?: Snippet;
	}
	import { copy } from 'svelte-copy';
	const { anchorname, language, filename, code, children }: CodeExampleProps = $props();

	const trimmedCode = code.trim();
	let buttonText = $state('Copy');
	function onCopy() {
		buttonText = 'Copied!';
		setTimeout(() => {
			buttonText = 'Copy';
		}, 2000);
	}
</script>

{#if anchorname !== null}
	<h2 id={anchorname}>{filename}</h2>
	<a href="#toc">(Back to top)</a>
{/if}

{#if children}
	{@render children()}
{/if}

<div class="code-example">
	<Highlight {language} code={trimmedCode} />
	<button
		class="copy"
		use:copy={{
			text: trimmedCode,
			onCopy
		}}>{buttonText}</button
	>
</div>

<style>
	.code-example {
		position: relative;
	}

	button.copy {
		font-family: inherit;
		padding: 4px 8px;
		border: 1.5px solid black;
		border-radius: 5px;
		cursor: pointer;
		transition: background-color 0.3s ease;
		position: absolute;
		top: 10px;
		/* the <pre> class has extra 50px margin so we adjust for that */
		right: 70px;
		font-size: 0.8em;
	}
	button.copy:hover {
		background-color: #f3f3f3;
	}
</style>
