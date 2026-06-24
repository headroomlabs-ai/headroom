/**
 * Example 03: Streaming Chat with Compression
 *
 * Use withHeadroom with streamText for real-time streaming responses.
 * Compression happens before the stream starts — the LLM sees fewer tokens.
 *
 * Run: npx tsx examples/03-streaming-chat.ts
 */
import { withHeadroom } from "headroom-ai/vercel-ai";
import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

// Simulate a long conversation with code review context
const codeReviewMessages = [
  {
    role: "system" as const,
    content: "You are a senior code reviewer. Be thorough but concise.",
  },
  {
    role: "user" as const,
    content: "Review this pull request diff",
  },
  {
    role: "assistant" as const,
    content: null,
    toolInvocations: [] as any[],
  },
  {
    role: "tool" as const,
    content: [
      {
        type: "tool-result" as const,
        toolCallId: "call_diff",
        toolName: "get_pr_diff",
        result: generateLargeDiff(),
      },
    ],
  },
  {
    role: "user" as const,
    content: "What are the most critical issues in this diff?",
  },
];

function generateLargeDiff(): string {
  const files = Array.from({ length: 15 }, (_, i) => {
    const lines = Array.from({ length: 20 }, (_, j) => {
      const prefix = j % 5 === 0 ? "+" : j % 7 === 0 ? "-" : " ";
      return `${prefix} ${j % 5 === 0 ? "// TODO: refactor this" : `const value${j} = process(input${j});`}`;
    });
    return `--- a/src/module${i}/handler.ts\n+++ b/src/module${i}/handler.ts\n@@ -1,20 +1,20 @@\n${lines.join("\n")}`;
  });
  return files.join("\n\n");
}

async function main() {
  const model = withHeadroom(openai("gpt-4o"));

  const result = streamText({
    model,
    messages: codeReviewMessages,
  });

  process.stdout.write("Review: ");
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }
  console.log("\n\nDone.");
}

main().catch(console.error);
