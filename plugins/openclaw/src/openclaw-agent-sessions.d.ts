declare module "openclaw/plugin-sdk/agent-sessions" {
  export class SessionManager {
    static open(target: unknown, cwd?: string): SessionManager;
    getBranch(): unknown[];
    branch(parentId: string): void;
    resetLeaf(): void;
    appendMessage(message: unknown): unknown;
  }
}
