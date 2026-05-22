import { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Cpu, MessageSquare, Power, RotateCw, Share2, Tag } from 'lucide-react';
import { Globe, peerMarkers } from '../components/Globe';
import { GlowButton, StatusDot } from '../components/ui';
import { listPeers, restartEngine, startEngine, stopEngine } from '../lib/api';
import { useApp } from '../lib/store';
import { cx, shortId, uptime } from '../lib/util';

export function Dashboard() {
  const engine = useApp((s) => s.engine);
  const setRoute = useApp((s) => s.setRoute);
  const [peerCount, setPeerCount] = useState(0);
  const [, tick] = useState(0);

  useEffect(() => {
    const load = () => listPeers().then((p) => setPeerCount(p.length));
    load();
    const id = setInterval(load, 5000);
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => {
      clearInterval(id);
      clearInterval(t);
    };
  }, []);

  const status = engine.status;
  const ready = engine.phase === 'ready';
  const connecting = engine.phase === 'starting';

  return (
    <div className="relative flex h-full flex-col items-center justify-center overflow-hidden">
      {/* Globe + radar rings, centred like the reference dashboard. */}
      <div className="relative flex items-center justify-center">
        <Globe size={440} active={ready} markers={peerMarkers(peerCount)} />
        {(ready || connecting) && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="absolute rounded-full border border-node/40"
                style={{
                  width: 120,
                  height: 120,
                  animation: `ripple 3s ease-out ${i * 1}s infinite`,
                }}
              />
            ))}
            <span className="absolute h-2.5 w-2.5 rounded-full bg-node shadow-node" />
          </div>
        )}
      </div>

      {/* Status line under the globe. */}
      <motion.div
        key={engine.phase}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-2 flex items-center gap-2 text-[14px]"
      >
        <StatusDot phase={engine.phase} size={9} />
        <span className={cx(ready ? 'text-ink-0' : 'text-ink-1')}>
          {ready ? 'Node online' : connecting ? 'Node connecting…' : 'Engine offline'}
        </span>
      </motion.div>

      {/* Engine controls */}
      <div className="mt-4 flex gap-2">
        {engine.phase === 'stopped' || engine.phase === 'error' ? (
          <GlowButton variant="node" onClick={() => startEngine()}>
            <Power size={15} /> Start node
          </GlowButton>
        ) : (
          <>
            <GlowButton onClick={() => restartEngine()}>
              <RotateCw size={14} /> Restart
            </GlowButton>
            <GlowButton onClick={() => stopEngine()}>
              <Power size={14} /> Stop
            </GlowButton>
          </>
        )}
        <GlowButton variant="primary" onClick={() => setRoute('chat')}>
          <MessageSquare size={15} /> Open chat
        </GlowButton>
      </div>

      {/* Stat strip */}
      <div className="absolute bottom-8 grid w-full max-w-3xl grid-cols-4 gap-3 px-8">
        <Stat icon={<Tag size={14} />} label="Model" value={status?.model?.name ?? '—'} mono={false} />
        <Stat icon={<Share2 size={14} />} label="Peers" value={String(peerCount)} />
        <Stat icon={<Cpu size={14} />} label="Node" value={shortId(status?.endpoint_id, 5, 4)} mono />
        <Stat
          icon={<Power size={14} />}
          label="Uptime"
          value={ready ? uptime(status?.started_at_unix) : '—'}
        />
      </div>
    </div>
  );
}

function Stat({
  icon,
  label,
  value,
  mono = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="glass rounded-2xl px-4 py-3">
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-ink-3">
        {icon}
        {label}
      </div>
      <div className={cx('mt-1 truncate text-[14px] text-ink-0', mono && 'font-mono text-[13px]')}>
        {value}
      </div>
    </div>
  );
}
