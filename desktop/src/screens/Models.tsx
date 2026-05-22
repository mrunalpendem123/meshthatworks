import { useEffect, useMemo, useState } from 'react';
import { Check, Download, HardDrive, Sparkles } from 'lucide-react';
import { CATALOG, CATEGORIES, COMPAT_LABEL } from '../lib/catalog';
import {
  downloadModel,
  listInstalledModels,
  onDownloadProgress,
  restartEngine,
  setActiveModel,
} from '../lib/api';
import type { DownloadProgress, InstalledModel } from '../lib/types';
import { useApp } from '../lib/store';
import { cx, formatBytes } from '../lib/util';
import { ProgressBar, Spinner } from '../components/ui';

export function Models({ onActivated }: { onActivated?: () => void }) {
  const setup = useApp((s) => s.setup);
  const [installed, setInstalled] = useState<InstalledModel[]>([]);
  const [filter, setFilter] = useState('All');
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<DownloadProgress | null>(null);

  const refresh = () => listInstalledModels().then(setInstalled);

  useEffect(() => {
    refresh();
    const un = onDownloadProgress((p) => {
      setProgress(p);
      if (p.phase === 'done') {
        setBusy(null);
        setProgress(null);
        refresh();
      }
    });
    return () => {
      un.then((f) => f());
    };
  }, []);

  const installedSet = useMemo(() => new Set(installed.map((m) => m.dirName)), [installed]);
  const activeDir = useMemo(() => installed.find((m) => m.isActive)?.dirName, [installed]);

  const shown = CATALOG.filter((m) => filter === 'All' || m.categories.includes(filter));

  async function activate(dirName: string) {
    setBusy(dirName);
    try {
      await setActiveModel(dirName);
      await refresh();
      if (setup?.swiftlmInstalled) await restartEngine().catch(() => {});
      onActivated?.();
    } finally {
      setBusy(null);
    }
  }

  async function download(repo: string, dirName: string) {
    setBusy(dirName);
    setProgress(null);
    try {
      await downloadModel(repo, dirName);
      await refresh();
      if (setup?.swiftlmInstalled) await restartEngine().catch(() => {});
      onActivated?.();
    } catch (e) {
      console.error(e);
      setBusy(null);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="mb-4 flex items-center gap-2">
        {CATEGORIES.map((c) => (
          <button
            key={c}
            onClick={() => setFilter(c)}
            className={cx(
              'rounded-full px-3.5 py-1.5 text-[12.5px] transition-colors',
              filter === c ? 'bg-white/10 text-ink-0' : 'text-ink-2 hover:text-ink-1',
            )}
          >
            {c}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-3 overflow-y-auto pr-1 lg:grid-cols-2">
        {shown.map((m) => {
          const isInstalled = installedSet.has(m.dirName);
          const isActive = activeDir === m.dirName;
          const isBusy = busy === m.dirName;
          const dl = isBusy && progress && progress.dirName === m.dirName;
          const pct = dl && progress.overallTotal > 0
            ? (progress.overallReceived / progress.overallTotal) * 100
            : 0;

          return (
            <div
              key={m.dirName}
              className={cx(
                'glass rounded-2xl p-4 transition-colors',
                isActive && 'ring-1 ring-node/40',
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    {m.priority && <Sparkles size={14} className="text-aurora-amber" />}
                    <h3 className="truncate text-[15px] font-semibold text-ink-0">{m.name}</h3>
                  </div>
                  <p className="mt-1 text-[12px] leading-snug text-ink-2">{m.note}</p>
                </div>
                <CompatBadge compat={m.compat} />
              </div>

              <div className="mt-3 flex items-center justify-between">
                <div className="flex items-center gap-3 text-[11.5px] text-ink-3">
                  <span className="flex items-center gap-1">
                    <HardDrive size={12} />
                    {m.sizeGb} GB
                  </span>
                  <span className="font-mono">{m.arch}</span>
                </div>

                {isActive ? (
                  <span className="pill pill-node px-3 py-1.5 text-[12px]">
                    <Check size={13} /> Active
                  </span>
                ) : isBusy && !dl ? (
                  <span className="pill px-3 py-1.5 text-[12px] text-ink-1">
                    <Spinner size={12} /> Working
                  </span>
                ) : isInstalled ? (
                  <button
                    onClick={() => activate(m.dirName)}
                    className="pill px-3 py-1.5 text-[12px] text-ink-0"
                  >
                    Use this model
                  </button>
                ) : (
                  <button
                    onClick={() => download(m.hfRepo, m.dirName)}
                    className="pill px-3 py-1.5 text-[12px] text-ink-0"
                  >
                    <Download size={13} /> Download
                  </button>
                )}
              </div>

              {dl && (
                <div className="mt-3">
                  <ProgressBar value={pct} />
                  <div className="mt-1.5 flex justify-between text-[11px] text-ink-3">
                    <span className="truncate font-mono">
                      {progress.phase === 'listing' ? 'Listing files…' : progress.file.split('/').pop()}
                    </span>
                    <span>
                      {formatBytes(progress.overallReceived)} / {formatBytes(progress.overallTotal)}
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CompatBadge({ compat }: { compat: keyof typeof COMPAT_LABEL }) {
  const tone =
    compat === 'recommended'
      ? 'text-node border-node/30 bg-node/10'
      : compat === 'tight'
        ? 'text-aurora-amber border-aurora-amber/30 bg-aurora-amber/10'
        : 'text-ink-2 border-white/10 bg-white/5';
  return (
    <span className={cx('shrink-0 rounded-full border px-2 py-0.5 text-[10.5px]', tone)}>
      {COMPAT_LABEL[compat]}
    </span>
  );
}
