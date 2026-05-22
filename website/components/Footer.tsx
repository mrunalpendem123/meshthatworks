import { Wordmark } from './Brand';

const REPO = 'https://github.com/mrunalpendem123/meshthatworks';

export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="mx-auto max-w-page px-6 pb-12 pt-8">
      <div className="glass rounded-2xl px-7 py-8">
        <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <Wordmark />
          <div className="flex flex-wrap gap-x-6 gap-y-2 text-[13px]">
            <a href={REPO} className="text-ink-2 transition-colors hover:text-ink-0">GitHub</a>
            <a href={`${REPO}#readme`} className="text-ink-2 transition-colors hover:text-ink-0">Docs</a>
            <a href={`${REPO}/blob/master/docs/ARCHITECTURE.md`} className="text-ink-2 transition-colors hover:text-ink-0">Architecture</a>
            <a href={`${REPO}/blob/master/docs/BASELINES.md`} className="text-ink-2 transition-colors hover:text-ink-0">Benchmarks</a>
            <a href={`${REPO}/blob/master/LICENSE`} className="text-ink-2 transition-colors hover:text-ink-0">MIT License</a>
          </div>
        </div>
        <div className="mt-7 flex flex-col gap-2 border-t border-line pt-5 text-[12px] text-ink-3 md:flex-row md:items-center md:justify-between">
          <div>© {year} MeshThatWorks contributors. MIT.</div>
          <div>
            Built with{' '}
            <a className="text-ink-2 hover:text-ink-0" href="https://www.iroh.computer">iroh</a> ·{' '}
            <a className="text-ink-2 hover:text-ink-0" href="https://github.com/SharpAI/SwiftLM">SwiftLM</a> ·{' '}
            <a className="text-ink-2 hover:text-ink-0" href="https://github.com/ml-explore/mlx-swift">mlx-swift</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
