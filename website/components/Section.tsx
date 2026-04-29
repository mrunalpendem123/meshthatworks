import type { ReactNode } from 'react';

export function Section({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <section id={id} className="max-w-page mx-auto px-6">
      {children}
    </section>
  );
}
