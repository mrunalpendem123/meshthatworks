// Thin typed wrappers over the Tauri command + event bridge. Tauri converts
// camelCase JS argument keys to the Rust snake_case parameter names for us.

import { Channel, invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type {
  ChatEvent,
  ChatMessage,
  DownloadProgress,
  EngineStatus,
  InstalledModel,
  NodeStatus,
  PeerInfo,
  SetupState,
} from './types';

export const getSetupState = () => invoke<SetupState>('get_setup_state');
export const listInstalledModels = () => invoke<InstalledModel[]>('list_installed_models');
export const setActiveModel = (dirName: string) => invoke<string>('set_active_model', { dirName });
export const downloadModel = (repo: string, dirName: string) =>
  invoke<void>('download_model', { repo, dirName });

export const startEngine = () => invoke<void>('start_engine');
export const stopEngine = () => invoke<void>('stop_engine');
export const restartEngine = () => invoke<void>('restart_engine');
export const nodeStatus = () => invoke<NodeStatus>('node_status');

export const listPeers = () => invoke<PeerInfo[]>('list_peers');
export const pairStart = () => invoke<void>('pair_start');
export const pairCancel = () => invoke<void>('pair_cancel');
export const joinMesh = (invite: string) => invoke<void>('join_mesh', { invite });

export const runSetup = () => invoke<void>('run_setup');
export const openUrl = (url: string) => invoke<void>('open_url', { url });

export async function chatStream(
  messages: ChatMessage[],
  opts: { maxTokens?: number; temperature?: number },
  onEvent: (e: ChatEvent) => void,
): Promise<void> {
  const channel = new Channel<ChatEvent>();
  channel.onmessage = onEvent;
  await invoke('chat_stream', {
    messages,
    maxTokens: opts.maxTokens,
    temperature: opts.temperature,
    onEvent: channel,
  });
}

// ── event subscriptions ──────────────────────────────────────────────
export const onEngineStatus = (cb: (s: EngineStatus) => void): Promise<UnlistenFn> =>
  listen<EngineStatus>('engine-status', (e) => cb(e.payload));
export const onEngineLog = (cb: (line: string) => void): Promise<UnlistenFn> =>
  listen<string>('engine-log', (e) => cb(e.payload));
export const onDownloadProgress = (cb: (p: DownloadProgress) => void): Promise<UnlistenFn> =>
  listen<DownloadProgress>('download-progress', (e) => cb(e.payload));
export const onSetupLog = (cb: (line: string) => void): Promise<UnlistenFn> =>
  listen<string>('setup-log', (e) => cb(e.payload));
export const onSetupStatus = (
  cb: (s: { phase: string; message: string }) => void,
): Promise<UnlistenFn> => listen<{ phase: string; message: string }>('setup-status', (e) => cb(e.payload));
export const onPairInvite = (cb: (invite: string) => void): Promise<UnlistenFn> =>
  listen<string>('pair-invite', (e) => cb(e.payload));
export const onPairLog = (cb: (line: string) => void): Promise<UnlistenFn> =>
  listen<string>('pair-log', (e) => cb(e.payload));
export const onPairSuccess = (cb: (line: string) => void): Promise<UnlistenFn> =>
  listen<string>('pair-success', (e) => cb(e.payload));
