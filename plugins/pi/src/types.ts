export const POLICY_VERSION = "v1";

export interface ContextMessage {
  role: string;
}

export interface Candidate {
  key: string;
  toolCallId: string;
  toolName: string;
  originalText: string;
}

export interface PreparedEntry extends Candidate {
  compressedText: string;
  ccrHashes: string[];
  retrievals: Map<string, string>;
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  originalBytes: number;
  compressedBytes: number;
  sizeBytes: number;
  originalSha256: string;
  policyVersion: string;
  createdAt: number;
  lastAccessedAt: number;
}
