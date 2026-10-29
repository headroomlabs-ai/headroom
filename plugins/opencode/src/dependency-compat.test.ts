import { tool } from "@opencode-ai/plugin";
import { Effect, FileSystem, Path } from "effect";
import { Primitive } from "effect/unstable/cli";
import { describe, expect, it } from "vitest";

function parseConfig(content: string): Promise<unknown> {
  const fs = FileSystem.makeNoop({
    exists: () => Effect.succeed(true),
    stat: () => Effect.succeed({ type: "File" } as FileSystem.File.Info),
    readFileString: () => Effect.succeed(content),
  });
  return Effect.runPromise(
    Primitive.fileParse({ format: "toml" }).parse("/headroom/config.toml").pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.provide(Path.layer),
    ),
  );
}

describe("patched OpenCode dependency compatibility", () => {
  it("builds and validates a tool with the upstream Zod schema interface", () => {
    const schema = tool.schema.object({
      hash: tool.schema.string().regex(/^[a-f0-9]{24}$/i),
    });
    const retrieve = tool({
      description: "Retrieve an original tool result",
      args: schema.shape,
      async execute(args) {
        return args.hash;
      },
    });
    expect(schema.parse({ hash: "a".repeat(24) })).toEqual({ hash: "a".repeat(24) });
    expect(schema.safeParse({ hash: "invalid" }).success).toBe(false);
    expect(retrieve.args.hash).toBe(schema.shape.hash);
  });

  it("parses typed TOML through Effect's actual file parser", async () => {
    const content = `
name = "headroom"
enabled = true
ports = [8787, 8788]
[proxy]
host = "127.0.0.1"
ratio = 0.5
[[models]]
name = "coding"
[[models]]
name = "review"
`;
    await expect(parseConfig(content)).resolves.toEqual({
      name: "headroom",
      enabled: true,
      ports: [8787, 8788],
      proxy: { host: "127.0.0.1", ratio: 0.5 },
      models: [{ name: "coding" }, { name: "review" }],
    });
  });

  it("keeps malformed TOML on Effect's parse-error channel", async () => {
    await expect(parseConfig("[unclosed")).rejects.toThrow("Failed to parse '.toml'");
  });
});
