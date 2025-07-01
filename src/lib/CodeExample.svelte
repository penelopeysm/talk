<script lang="ts">
	import Highlight from 'svelte-highlight';
    import type { LanguageType } from 'svelte-highlight/languages';
	interface CodeExampleProps {
		anchorname: string;
		language: LanguageType<string>;
		filename: string;
		code: string;
        description: any;
	}
	const { anchorname, language, filename, code, description }: CodeExampleProps = $props();
</script>

<h2 id={anchorname}>{filename}</h2>
<a href="#toc">(Back to top)</a>

{@render description()}

<Highlight {language} {code} />
