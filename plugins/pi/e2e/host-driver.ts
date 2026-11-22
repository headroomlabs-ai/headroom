import { randomUUID } from "node:crypto";
import { rename, writeFile } from "node:fs/promises";
import {
  createAssistantMessageEventStream,
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Model,
  type TextContent,
  type ToolCall,
  type ToolResultMessage,
} from "@earendil-works/pi-ai";
import type {
  ExtensionAPI,
  ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const providerA = "headroom-e2e-a";
const providerB = "headroom-e2e-b";
const apiA = "headroom-e2e-a-api" as Api;
const apiB = "headroom-e2e-b-api" as Api;
const resultPath = process.env.HEADROOM_HOST_E2E_RESULT;
const runNonce = randomUUID();
const rawOutputs = Array.from({ length: 3 }, (_, fixtureIndex) =>
  Array.from(
    { length: 350 },
    (_, lineIndex) =>
      `HEADROOM_E2E_RAW_${fixtureIndex} run=${runNonce} ${String(lineIndex).padStart(4, "0")} INFO worker=${lineIndex % 16} request=${fixtureIndex}-${lineIndex} completed status=200 latency_ms=${20 + (lineIndex % 53)} cache=${lineIndex % 7 === 0 ? "hit" : "miss"}`,
  ).join("\n"),
);

const baseModel = {
  name: "Headroom E2E",
  baseUrl: "http://127.0.0.1/headroom-e2e",
  reasoning: false,
  input: ["text"] as ("text" | "image")[],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 128_000,
  maxTokens: 2_048,
};
const modelA: Model<Api> = {
  ...baseModel,
  id: "model-a",
  provider: providerA,
  api: apiA,
};
const modelB: Model<Api> = {
  ...baseModel,
  id: "model-b",
  provider: providerB,
  api: apiB,
};

let phase: 0 | 1 | 2 = 0;
let providerEvidence:
  | {
      provider: string;
      toolResultCount: number;
      firstResultCompressed: boolean;
      recentResultsRaw: boolean;
    }
  | undefined;
const streamDiagnostics: Array<{
  provider: string;
  prompt: string;
  roles: string[];
  toolResultCount: number;
}> = [];

function messageText(message: ToolResultMessage): string {
  return message.content
    .filter((content): content is TextContent => content.type === "text")
    .map((content) => content.text)
    .join("\n");
}

function userText(context: Context): string {
  const userMessages = [...context.messages]
    .reverse()
    .filter((message) => message.role === "user")
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : message.content
            .filter((content): content is TextContent => content.type === "text")
            .map((content) => content.text)
            .join("\n"),
    );
  return (
    userMessages.find((text) => /\bphase-(?:one|two)\b/.test(text)) ??
    userMessages[0] ??
    ""
  );
}

function outputMessage(model: Model<Api>): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

function finishText(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
  text: string,
): void {
  const content: TextContent = { type: "text", text };
  output.content.push(content);
  output.stopReason = "stop";
  stream.push({ type: "start", partial: output });
  stream.push({ type: "text_start", contentIndex: 0, partial: output });
  stream.push({ type: "text_delta", contentIndex: 0, delta: text, partial: output });
  stream.push({ type: "text_end", contentIndex: 0, content: text, partial: output });
  stream.push({ type: "done", reason: "stop", message: output });
  stream.end();
}

function finishToolCalls(
  stream: AssistantMessageEventStream,
  output: AssistantMessage,
): void {
  const calls: ToolCall[] = rawOutputs.map((_, index) => ({
    type: "toolCall",
    id: `headroom-e2e-call-${index}`,
    name: "headroom_e2e_blob",
    arguments: { index },
  }));
  output.stopReason = "toolUse";
  stream.push({ type: "start", partial: output });
  for (const [contentIndex, toolCall] of calls.entries()) {
    output.content.push(toolCall);
    stream.push({ type: "toolcall_start", contentIndex, partial: output });
    stream.push({
      type: "toolcall_end",
      contentIndex,
      toolCall,
      partial: output,
    });
  }
  stream.push({ type: "done", reason: "toolUse", message: output });
  stream.end();
}

function deterministicStream(
  model: Model<Api>,
  context: Context,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  queueMicrotask(() => {
    const output = outputMessage(model);
    const prompt = userText(context);
    const toolResults = context.messages.filter(
      (message): message is ToolResultMessage => message.role === "toolResult",
    );
    streamDiagnostics.push({
      provider: model.provider,
      prompt,
      roles: context.messages.map((message) => message.role),
      toolResultCount: toolResults.length,
    });

    if (prompt.includes("phase-one") && toolResults.length === 0) {
      finishToolCalls(stream, output);
      return;
    }
    if (prompt.includes("phase-two")) {
      const texts = toolResults.map(messageText);
      providerEvidence = {
        provider: model.provider,
        toolResultCount: texts.length,
        firstResultCompressed:
          texts.length === 3 &&
          !texts[0]?.startsWith("HEADROOM_E2E_RAW_0") &&
          /Retrieve more: hash=|<<ccr:|<headroom:/.test(texts[0] ?? ""),
        recentResultsRaw:
          texts[1] === rawOutputs[1] && texts[2] === rawOutputs[2],
      };
      finishText(stream, output, "phase-two-complete");
      return;
    }
    finishText(stream, output, "phase-one-complete");
  });
  return stream;
}

function rawSessionEvidence(ctx: ExtensionContext): {
  exactRawResults: boolean;
  modelSwitchApplied: boolean;
} {
  const entries = ctx.sessionManager.getBranch();
  const toolResultTexts = entries
    .filter((entry) => entry.type === "message" && entry.message.role === "toolResult")
    .map((entry) =>
      entry.type === "message" && entry.message.role === "toolResult"
        ? messageText(entry.message)
        : "",
    );
  return {
    exactRawResults: rawOutputs.every(
      (expected, index) => toolResultTexts[index] === expected,
    ),
    modelSwitchApplied: ctx.model?.provider === providerB,
  };
}

async function recordResult(
  ctx: ExtensionContext,
  error?: unknown,
): Promise<void> {
  const evidence = {
    ok: false,
    providerEvidence,
    streamDiagnostics,
    ...rawSessionEvidence(ctx),
    error: error instanceof Error ? error.message : error ? String(error) : undefined,
  };
  evidence.ok =
    evidence.error === undefined &&
    evidence.providerEvidence?.provider === providerB &&
    evidence.providerEvidence.toolResultCount === 3 &&
    evidence.providerEvidence.firstResultCompressed &&
    evidence.providerEvidence.recentResultsRaw &&
    evidence.exactRawResults &&
    evidence.modelSwitchApplied;
  if (resultPath) {
    const temporaryResultPath = `${resultPath}.tmp`;
    await writeFile(temporaryResultPath, JSON.stringify(evidence));
    await rename(temporaryResultPath, resultPath);
  }
  ctx.ui.notify(evidence.ok ? "HEADROOM_E2E_PASS" : "HEADROOM_E2E_FAIL", evidence.ok ? "info" : "error");
  ctx.shutdown();
}

export default function headroomHostDriver(pi: ExtensionAPI): void {
  pi.registerProvider(providerA, {
    baseUrl: baseModel.baseUrl,
    apiKey: "e2e",
    api: apiA,
    models: [modelA],
    streamSimple: deterministicStream,
  });
  pi.registerProvider(providerB, {
    baseUrl: baseModel.baseUrl,
    apiKey: "e2e",
    api: apiB,
    models: [modelB],
    streamSimple: deterministicStream,
  });
  pi.registerTool({
    name: "headroom_e2e_blob",
    label: "Headroom E2E blob",
    description: "Return deterministic host integration output",
    parameters: Type.Object({ index: Type.Integer({ minimum: 0, maximum: 2 }) }),
    async execute(_toolCallId, params) {
      return {
        content: [{ type: "text" as const, text: rawOutputs[params.index] ?? "" }],
        details: { index: params.index },
      };
    },
  });
  pi.registerCommand("headroom-e2e", {
    description: "Run the deterministic Headroom host integration gate",
    handler: async (_args, ctx) => {
      try {
        phase = 1;
        if (!(await pi.setModel(modelA))) throw new Error("could not select model A");
        pi.sendUserMessage("phase-one");
      } catch (error) {
        await recordResult(ctx, error);
      }
    },
  });
  pi.on("agent_end", async (_event, ctx) => {
    try {
      if (phase === 1) {
        phase = 2;
        await new Promise((resolve) => setTimeout(resolve, 8_000));
        if (!(await pi.setModel(modelB))) throw new Error("could not select model B");
        pi.sendUserMessage("phase-two", { deliverAs: "followUp" });
        return;
      }
      if (phase === 2) {
        phase = 0;
        await recordResult(ctx);
      }
    } catch (error) {
      phase = 0;
      await recordResult(ctx, error);
    }
  });
}
