// Shapes shared with the Rust side. SetupState / InstalledModel / PeerInfo /
// DownloadProgress use camelCase (the Rust structs rename to camelCase). The
// /status payload is forwarded verbatim from mtw-api, so it stays snake_case.

export type EnginePhase = 'stopped' | 'starting' | 'ready' | 'error';

export interface ModelInfo {
  name: string;
  num_layers: number;
  hidden_size: number;
}

export interface NodeStatus {
  endpoint_id: string;
  proxy_url: string;
  upstream_url: string;
  alpns: string[];
  model: ModelInfo;
  started_at_unix: number;
  version: string;
}

export interface EngineStatus {
  phase: EnginePhase;
  healthy: boolean;
  message: string;
  status: NodeStatus | null;
}

export interface SetupState {
  metalAvailable: boolean;
  swiftlmInstalled: boolean;
  swiftlmPath: string;
  modelInstalled: boolean;
  activeModel: string | null;
  activeModelName: string | null;
  installedCount: number;
  depsDir: string;
  modelsDir: string;
  ready: boolean;
}

export interface InstalledModel {
  dirName: string;
  path: string;
  sizeBytes: number;
  isActive: boolean;
}

export interface PeerInfo {
  id: string;
  pairedAt: number;
}

export interface DownloadProgress {
  dirName: string;
  file: string;
  fileIndex: number;
  fileCount: number;
  received: number;
  total: number;
  overallReceived: number;
  overallTotal: number;
  phase: 'listing' | 'downloading' | 'done' | 'error';
}

export type ChatEvent =
  | { type: 'delta'; content: string }
  | { type: 'done' }
  | { type: 'error'; message: string };

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export type Compat = 'recommended' | 'tight' | 'needsBigger';

export interface CatalogModel {
  hfRepo: string;
  dirName: string;
  name: string;
  sizeGb: number;
  arch: string;
  compat: Compat;
  categories: string[];
  note: string;
  priority?: boolean;
}
