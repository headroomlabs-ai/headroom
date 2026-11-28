export type HeadroomSelection = { requestId: string; side: 'original' | 'compressed' | 'diff'; message: number; page: number };
export type HeadroomState = {
  startNotice?: string;
  owner: number;
  compressionEnabled: boolean | null;
  timeWindow: 'all' | '15m' | '1h' | '24h';
  controlNotice: string;
  controlPending: boolean;
  showDetails: boolean;
  sessionId: string;
  resumeId?: string | null;
  baseUrl: string;
  open: boolean;
  band: boolean;
  tab: 'overview' | 'requests' | 'inspect';
  connection: 'setup' | 'live' | 'offline';
  notice: string;
  updatedAt: number | null;
  context: { tokens?: number; window?: number; percent?: number } | null;
  // Validated schema-1 JSON responses. No provider credentials or raw global feed.
  summary: Record<string, unknown> | null;
  detail: Record<string, unknown> | null;
  selection: HeadroomSelection | null;
  listPage: number;
  changedOnly: boolean;
  textPage: number;
};
declare module 'claude-code' {
  interface PluginState { "headroom-sidebar": { model: HeadroomState }; }
}
