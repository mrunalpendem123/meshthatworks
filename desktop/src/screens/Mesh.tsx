import { useEffect, useState } from 'react';
import { Check, Copy, Link2, Plus, Radio, X } from 'lucide-react';
import { Globe, peerMarkers } from '../components/Globe';
import { GlowButton, Spinner, StatusDot } from '../components/ui';
import {
  discoveredNodes,
  joinMesh,
  listPeers,
  onPairInvite,
  onPairLog,
  onPairSuccess,
  pairCancel,
  pairStart,
  setSplitMode,
  splitStatus,
} from '../lib/api';
import type { DiscoveredNode } from '../lib/api';
import type { PeerInfo } from '../lib/types';
import { useApp } from '../lib/store';
import { shortId } from '../lib/util';

export function Mesh() {
  const engine = useApp((s) => s.engine);
  const nodeId = engine.status?.endpoint_id;
  const [peers, setPeers] = useState<PeerInfo[]>([]);
  const [pairing, setPairing] = useState(false);
  const [invite, setInvite] = useState('');
  const [pairLog, setPairLog] = useState('');
  const [copied, setCopied] = useState(false);
  const [joinInput, setJoinInput] = useState('');
  const [joining, setJoining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [splitLabel, setSplitLabel] = useState('off');
  const [discovered, setDiscovered] = useState<DiscoveredNode[]>([]);

  const refresh = () => listPeers().then(setPeers);

  async function applySplit(role: 'off' | 'head' | 'worker', peer?: string) {
    setError(null);
    setSplitLabel(role === 'head' ? `head→${shortId(peer, 6, 4)}` : role);
    try {
      await setSplitMode(role, peer);
      setTimeout(() => splitStatus().then(setSplitLabel).catch(() => {}), 1800);
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    refresh();
    splitStatus().then(setSplitLabel).catch(() => {});
    discoveredNodes().then(setDiscovered).catch(() => {});
    const poll = setInterval(() => discoveredNodes().then(setDiscovered).catch(() => {}), 5000);
    const a = onPairInvite((inv) => setInvite(inv));
    const b = onPairLog((line) => setPairLog(line));
    const c = onPairSuccess(() => {
      setPairing(false);
      setInvite('');
      refresh();
    });
    return () => {
      clearInterval(poll);
      [a, b, c].forEach((p) => p.then((f) => f()));
    };
  }, []);

  async function startPairing() {
    setError(null);
    setInvite('');
    setPairLog('Generating invite…');
    setPairing(true);
    try {
      await pairStart();
    } catch (e) {
      setError(String(e));
      setPairing(false);
    }
  }

  async function cancelPairing() {
    await pairCancel().catch(() => {});
    setPairing(false);
    setInvite('');
  }

  async function copyInvite() {
    await navigator.clipboard.writeText(invite);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  async function doJoin() {
    if (!joinInput.trim()) return;
    setJoining(true);
    setError(null);
    try {
      await joinMesh(joinInput.trim());
      setJoinInput('');
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setJoining(false);
    }
  }

  return (
    <div className="mx-auto grid h-full w-full max-w-5xl grid-cols-1 gap-6 px-8 py-2 lg:grid-cols-[1fr_360px]">
      {/* left: this node + peers */}
      <div className="flex min-h-0 flex-col">
        <div className="glass rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 text-[11px] uppercase tracking-wider text-ink-3">
                <Radio size={13} /> This node
              </div>
              <div className="mt-1 font-mono text-[14px] text-ink-0">{shortId(nodeId, 10, 6)}</div>
            </div>
            <div className="pill px-3 py-1.5 text-[12.5px] text-ink-1">
              <StatusDot phase={engine.phase} /> {engine.phase === 'ready' ? 'online' : engine.phase}
            </div>
          </div>
        </div>

        {/* split-mode control */}
        <div className="glass mt-3 rounded-2xl p-4">
          <div className="flex items-center justify-between">
            <div className="text-[12.5px] text-ink-2">
              Split mode: <span className="font-mono text-ink-0">{splitLabel}</span>
            </div>
            <div className="flex gap-2">
              <button onClick={() => applySplit('worker')} className="pill px-3 py-1 text-[12px] text-ink-1">
                Be worker
              </button>
              <button onClick={() => applySplit('off')} className="pill px-3 py-1 text-[12px] text-ink-2">
                Single
              </button>
            </div>
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-ink-3">
            Run one big model across both Macs: on the other Mac click <b>Be worker</b>; here, click
            <b> Split&nbsp;here</b> next to it. Then chat — the model splits across both over iroh.
            (Engine restarts; both need the same model.)
          </p>
        </div>

        {/* discovered swarm nodes (live, over gossip — no invite codes) */}
        <div className="mt-4 mb-2 flex items-center justify-between">
          <h2 className="text-[15px] font-semibold text-ink-0">
            Online now <span className="text-ink-3">· {discovered.length}</span>
          </h2>
          <span className="text-[11px] text-ink-3">discovered · no codes</span>
        </div>
        <div className="space-y-2">
          {discovered.length === 0 ? (
            <p className="text-[12px] text-ink-3">
              No other nodes online yet. Pair once to join the swarm — after that, everyone on it
              shows up here automatically.
            </p>
          ) : (
            discovered.map((d) => (
              <div key={d.endpointId} className="glass flex items-center gap-3 rounded-xl px-4 py-3">
                <span className="h-2 w-2 rounded-full bg-node shadow-node" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[13px] text-ink-0">{d.name}</div>
                  <div className="truncate text-[11px] text-ink-3">{d.model}</div>
                </div>
                <button
                  onClick={() => applySplit('head', d.endpointId)}
                  className="pill px-2.5 py-1 text-[11px] text-node"
                  title="Split a model across both, driven from here"
                >
                  Split here
                </button>
              </div>
            ))
          )}
        </div>

        <div className="mt-5 mb-3 flex items-center justify-between">
          <h2 className="text-[15px] font-semibold text-ink-0">
            Paired devices <span className="text-ink-3">· {peers.length}</span>
          </h2>
        </div>

        <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
          {peers.length === 0 ? (
            <div className="glass flex flex-col items-center rounded-2xl px-6 py-10 text-center">
              <Link2 size={26} className="text-ink-3" />
              <p className="mt-3 text-[13.5px] text-ink-1">No devices paired yet</p>
              <p className="mt-1 max-w-xs text-[12px] text-ink-3">
                Pair a second Mac to split big models across both — each holds only a slice.
              </p>
            </div>
          ) : (
            peers.map((p) => (
              <div key={p.id} className="glass flex items-center gap-3 rounded-xl px-4 py-3">
                <span className="h-2 w-2 rounded-full bg-node shadow-node" />
                <span className="flex-1 truncate font-mono text-[13px] text-ink-0">
                  {shortId(p.id, 12, 8)}
                </span>
                <button
                  onClick={() => applySplit('head', p.id)}
                  className="pill px-2.5 py-1 text-[11px] text-node"
                  title="Split a model across both Macs, driven from here"
                >
                  Split here
                </button>
                <span className="text-[11px] text-ink-3">
                  {new Date(p.pairedAt * 1000).toLocaleDateString()}
                </span>
              </div>
            ))
          )}
        </div>

        {error && (
          <p className="mt-3 rounded-lg bg-aurora-amber/10 px-3 py-2 text-[12px] text-aurora-amber">
            {error}
          </p>
        )}
      </div>

      {/* right: pair / join */}
      <div className="flex flex-col gap-4">
        <div className="relative flex items-center justify-center overflow-hidden rounded-2xl">
          <Globe size={300} active={engine.phase === 'ready'} markers={peerMarkers(peers.length)} />
        </div>

        {/* create invite */}
        <div className="glass rounded-2xl p-5">
          <h3 className="text-[14px] font-semibold text-ink-0">Add a device</h3>
          {!pairing ? (
            <>
              <p className="mt-1 text-[12px] text-ink-2">
                Create a one-time invite key, then enter it on the other Mac.
              </p>
              <GlowButton variant="node" onClick={startPairing} className="mt-3 w-full justify-center">
                <Plus size={15} /> Create invite key
              </GlowButton>
            </>
          ) : invite ? (
            <>
              <p className="mt-1 text-[12px] text-ink-2">Enter this on the other device:</p>
              <div className="mt-2 flex items-center gap-2 rounded-xl border border-node/30 bg-node/[0.06] p-2.5">
                <code className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-node">
                  {invite}
                </code>
                <button
                  onClick={copyInvite}
                  className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-white/10 text-ink-0"
                >
                  {copied ? <Check size={14} /> : <Copy size={13} />}
                </button>
              </div>
              <div className="mt-3 flex items-center justify-between text-[12px] text-ink-2">
                <span className="flex items-center gap-2">
                  <Spinner size={13} /> waiting for a device…
                </span>
                <button onClick={cancelPairing} className="flex items-center gap-1 text-ink-3 hover:text-ink-1">
                  <X size={13} /> cancel
                </button>
              </div>
            </>
          ) : (
            <div className="mt-3 flex items-center gap-2 text-[12.5px] text-ink-2">
              <Spinner size={14} /> {pairLog || 'Generating invite…'}
            </div>
          )}
        </div>

        {/* join */}
        <div className="glass rounded-2xl p-5">
          <h3 className="text-[14px] font-semibold text-ink-0">Join a mesh</h3>
          <p className="mt-1 text-[12px] text-ink-2">Have an invite key from another device?</p>
          <div className="input-glass mt-3 flex items-center px-1.5 py-1.5">
            <input
              value={joinInput}
              onChange={(e) => setJoinInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && doJoin()}
              placeholder="mtw-invite:…"
              className="min-w-0 flex-1 bg-transparent px-3 text-[12.5px] text-ink-0 placeholder:text-ink-3 focus:outline-none"
            />
            <button
              onClick={doJoin}
              disabled={joining || !joinInput.trim()}
              className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-white/10 text-ink-0 disabled:opacity-40"
            >
              {joining ? <Spinner size={13} /> : <Link2 size={14} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
