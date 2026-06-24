interface CodeBlockProps {
  code: string;
  lang?: string;
}

export function CodeBlock({ code }: CodeBlockProps) {
  return (
    <pre className="mt-3 p-3 text-xs rounded-lg bg-fd-muted text-fd-foreground overflow-x-auto font-mono whitespace-pre-wrap break-words">
      {code}
    </pre>
  );
}
