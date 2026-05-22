import { Loader2 } from 'lucide-react';
import type { ReactNode } from 'react';
import type { EnginePhase } from '../lib/types';
import { cx } from '../lib/util';

export function StatusDot({ phase, size = 8 }: { phase: EnginePhase; size?: number }) {
  const color =
    phase === 'ready'
      ? '#5ce8a6'
      : phase === 'starting'
        ? '#d8b46a'
        : phase === 'error'
          ? '#e8746a'
          : '#5c6378';
  return (
    <span
      className={cx('inline-block rounded-full', phase === 'starting' && 'animate-breathe')}
      style={{
        width: size,
        height: size,
        background: color,
        boxShadow: phase === 'ready' ? `0 0 8px ${color}` : undefined,
      }}
    />
  );
}

export function Spinner({ size = 16, className }: { size?: number; className?: string }) {
  return <Loader2 size={size} className={cx('animate-spin', className)} />;
}

export function ProgressBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={cx('h-1.5 w-full overflow-hidden rounded-full bg-white/8', className)}>
      <div
        className="h-full rounded-full bg-node transition-[width] duration-200 ease-out"
        style={{ width: `${Math.max(0, Math.min(100, value))}%`, boxShadow: '0 0 12px rgba(92,232,166,0.5)' }}
      />
    </div>
  );
}

export function IconButton({
  children,
  onClick,
  title,
  active,
}: {
  children: ReactNode;
  onClick?: () => void;
  title?: string;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={cx(
        'no-drag grid h-9 w-9 place-items-center rounded-full border transition-colors',
        active
          ? 'border-node/40 bg-node/10 text-node'
          : 'border-white/10 bg-white/[0.03] text-ink-1 hover:border-white/25 hover:text-ink-0',
      )}
    >
      {children}
    </button>
  );
}

export function GlowButton({
  children,
  onClick,
  disabled,
  variant = 'primary',
  className,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'ghost' | 'node';
  className?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cx(
        'no-drag px-5 py-2.5 text-sm',
        variant === 'primary' ? 'pill pill-primary' : variant === 'node' ? 'pill pill-node' : 'pill',
        disabled && 'cursor-not-allowed opacity-40',
        className,
      )}
    >
      {children}
    </button>
  );
}
