import { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { ArrowRight, Check, Cpu, Boxes, TriangleAlert } from 'lucide-react';
import { Globe } from '../components/Globe';
import { Wordmark } from '../components/Brand';
import { GlowButton, Spinner } from '../components/ui';
import { Models } from './Models';
import { joinMesh, onSetupLog, onSetupStatus, openUrl, runSetup, startEngine } from '../lib/api';
import { useApp } from '../lib/store';
import { cx } from '../lib/util';

export function Onboarding({
  refreshSetup,
  onEnter,
}: {
  refreshSetup: () => Promise<void>;
  onEnter: () => void;
}) {
  const setup = useApp((s) => s.setup);
  const [view, setView] = useState<'steps' | 'models'>('steps');
  const [preparing, setPreparing] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [invite, setInvite] = useState('');
  const [joining, setJoining] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const a = onSetupLog((line) => setLog((l) => [...l.slice(-300), line]));
    const b = onSetupStatus((s) => {
      if (s.phase === 'done') {
        setPreparing(false);
        refreshSetup();
      } else if (s.phase === 'error') {
        setPreparing(false);
        setError(s.message);
      }
    });
    return () => {
      a.then((f) => f());
      b.then((f) => f());
    };
  }, [refreshSetup]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [log]);

  async function prepare() {
    setError(null);
    setPreparing(true);
    setLog([]);
    try {
      await runSetup();
      await refreshSetup();
    } catch (e) {
      setError(String(e));
    } finally {
      setPreparing(false);
    }
  }

  async function enter() {
    await startEngine().catch(() => {});
    onEnter();
  }

  async function join() {
    if (!invite.trim()) return;
    setJoining(true);
    setError(null);
    try {
      await joinMesh(invite.trim());
      setInvite('');
    } catch (e) {
      setError(String(e));
    } finally {
      setJoining(false);
    }
  }

  if (view === 'models') {
    return (
      <div className="mx-auto flex h-full w-full max-w-4xl flex-col px-8 pb-8 pt-2">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-ink-0">Choose a model</h2>
            <p className="text-[13px] text-ink-2">
              MoE models stream from your SSD. Pick a small one first to verify everything works.
            </p>
          </div>
          <GlowButton onClick={() => setView('steps')}>Back to setup</GlowButton>
        </div>
        <div className="min-h-0 flex-1">
          <Models onActivated={() => refreshSetup()} />
        </div>
        {setup?.ready && (
          <div className="mt-4 flex justify-end">
            <GlowButton variant="primary" onClick={enter}>
              Enter the mesh <ArrowRight size={16} />
            </GlowButton>
          </div>
        )}
      </div>
    );
  }

  const metalOk = setup?.metalAvailable ?? true;
  const engineOk = setup?.swiftlmInstalled ?? false;
  const modelOk = (setup?.modelInstalled ?? false) && !!setup?.activeModel;

  return (
    <div className="relative flex h-full items-center justify-center overflow-hidden px-8">
      <motion.div
        initial={{ opacity: 0, x: -30 }}
        animate={{ opacity: 0.8, x: 0 }}
        transition={{ duration: 1.2 }}
        className="pointer-events-none absolute -left-32 top-1/2 -translate-y-1/2 opacity-70"
      >
        <Globe size={520} speed={0.002} active={engineOk} />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="glass relative z-10 w-full max-w-md rounded-3xl p-7"
      >
        <Wordmark />
        <h1 className="mt-5 text-2xl font-semibold tracking-tight text-ink-0">Set up this device</h1>
        <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">
          Two steps and your Mac becomes a node. Everything runs locally — no cloud, no account.
        </p>

        {!metalOk ? (
          <div className="mt-5 rounded-2xl border border-aurora-amber/30 bg-aurora-amber/10 p-4">
            <div className="flex items-center gap-2 text-aurora-amber">
              <TriangleAlert size={16} />
              <span className="text-sm font-medium">Xcode + Metal Toolchain needed</span>
            </div>
            <p className="mt-2 text-[12.5px] leading-relaxed text-ink-1">
              SwiftLM compiles Metal shaders. Install Xcode, then run in Terminal:
            </p>
            <code className="mt-2 block rounded-lg bg-black/40 px-3 py-2 font-mono text-[11.5px] text-node">
              xcodebuild -downloadComponent MetalToolchain
            </code>
            <div className="mt-3 flex gap-2">
              <GlowButton onClick={() => openUrl('https://apps.apple.com/app/xcode/id497799835')}>
                Get Xcode
              </GlowButton>
              <GlowButton variant="node" onClick={refreshSetup}>
                Re-check
              </GlowButton>
            </div>
          </div>
        ) : (
          <div className="mt-6 space-y-3">
            <Step
              n={1}
              Icon={Cpu}
              title="Prepare the engine"
              done={engineOk}
              detail={
                engineOk
                  ? 'SwiftLM is built and ready.'
                  : 'Builds SwiftLM (~30 min, once). Needs Xcode.'
              }
              action={
                !engineOk &&
                (preparing ? (
                  <span className="pill px-3 py-1.5 text-[12px] text-ink-1">
                    <Spinner size={12} /> Building…
                  </span>
                ) : (
                  <button onClick={prepare} className="pill px-3.5 py-1.5 text-[12px] text-ink-0">
                    Prepare
                  </button>
                ))
              }
            />

            <Step
              n={2}
              Icon={Boxes}
              title="Choose a model"
              done={modelOk}
              detail={modelOk ? (setup?.activeModelName ?? 'Model ready.') : 'Download an open MoE model.'}
              action={
                <button
                  onClick={() => setView('models')}
                  className="pill px-3.5 py-1.5 text-[12px] text-ink-0"
                >
                  {modelOk ? 'Change' : 'Choose'}
                </button>
              }
            />
          </div>
        )}

        {preparing && (
          <div
            ref={logRef}
            className="mt-4 max-h-32 overflow-y-auto rounded-xl border border-white/8 bg-black/40 p-3 font-mono text-[11px] leading-relaxed text-ink-2"
          >
            {log.length === 0 ? 'Starting…' : log.map((l, i) => <div key={i}>{l}</div>)}
          </div>
        )}

        {error && (
          <p className="mt-3 rounded-lg bg-aurora-amber/10 px-3 py-2 text-[12px] text-aurora-amber">
            {error}
          </p>
        )}

        <div className="mt-6">
          {setup?.ready ? (
            <GlowButton variant="primary" onClick={enter} className="w-full justify-center">
              Enter the mesh <ArrowRight size={16} />
            </GlowButton>
          ) : (
            <p className="text-center text-[12px] text-ink-3">Finish both steps to enter.</p>
          )}
        </div>

        {/* Magic-key style: join an existing mesh with an invite. */}
        <div className="mt-6 border-t border-white/8 pt-5">
          <p className="mb-2 text-center text-[12px] text-ink-3">Joining a friend's mesh?</p>
          <div className="input-glass flex items-center px-1.5 py-1.5">
            <input
              value={invite}
              onChange={(e) => setInvite(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && join()}
              placeholder="Paste invite key  ·  mtw-invite:…"
              className="min-w-0 flex-1 bg-transparent px-3 text-[13px] text-ink-0 placeholder:text-ink-3 focus:outline-none"
            />
            <button
              onClick={join}
              disabled={joining || !invite.trim()}
              className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-white/10 text-ink-0 disabled:opacity-40"
            >
              {joining ? <Spinner size={14} /> : <ArrowRight size={15} />}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

function Step({
  n,
  Icon,
  title,
  detail,
  done,
  action,
}: {
  n: number;
  Icon: typeof Cpu;
  title: string;
  detail: string;
  done: boolean;
  action?: React.ReactNode;
}) {
  return (
    <div
      className={cx(
        'flex items-center gap-3 rounded-2xl border p-3.5 transition-colors',
        done ? 'border-node/30 bg-node/[0.06]' : 'border-white/8 bg-white/[0.02]',
      )}
    >
      <div
        className={cx(
          'grid h-9 w-9 shrink-0 place-items-center rounded-xl',
          done ? 'bg-node/15 text-node' : 'bg-white/5 text-ink-2',
        )}
      >
        {done ? <Check size={17} /> : <Icon size={16} />}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-[13.5px] font-medium text-ink-0">
          <span className="text-ink-3">{n}.</span> {title}
        </div>
        <p className="truncate text-[12px] text-ink-2">{detail}</p>
      </div>
      {action}
    </div>
  );
}
