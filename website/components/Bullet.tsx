import type { ReactNode } from 'react';

export function Bullet({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="text-muted shrink-0">[*]</span>
      <div>
        <span className="text-fg font-semibold mr-2">{title}</span>
        <span className="text-fgDim">{children}</span>
      </div>
    </div>
  );
}
