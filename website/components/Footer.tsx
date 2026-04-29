const REPO = 'https://github.com/mrunalpendem123/meshthatworks';

export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer>
      <div className="max-w-page mx-auto px-6 py-10 grid grid-cols-2 md:grid-cols-5 gap-3 text-sm border-b border-line">
        <a href={REPO} className="text-fgDim hover:text-fg !border-b-0">
          GitHub
        </a>
        <a href={`${REPO}#readme`} className="text-fgDim hover:text-fg !border-b-0">
          Docs
        </a>
        <a href={`${REPO}/blob/master/docs/ARCHITECTURE.md`} className="text-fgDim hover:text-fg !border-b-0">
          Architecture
        </a>
        <a href={`${REPO}/blob/master/docs/BASELINES.md`} className="text-fgDim hover:text-fg !border-b-0">
          Benchmarks
        </a>
        <a href={`${REPO}/blob/master/LICENSE`} className="text-fgDim hover:text-fg !border-b-0">
          MIT License
        </a>
      </div>
      <div className="max-w-page mx-auto px-6 py-6 flex items-center justify-between text-xs text-muted">
        <div>© {year} MeshThatWorks contributors. MIT.</div>
        <div>
          Built with{' '}
          <a className="text-fgDim" href="https://www.iroh.computer">
            iroh
          </a>{' '}
          ·{' '}
          <a className="text-fgDim" href="https://github.com/SharpAI/SwiftLM">
            SwiftLM
          </a>{' '}
          ·{' '}
          <a className="text-fgDim" href="https://github.com/ml-explore/mlx-swift">
            mlx-swift
          </a>
        </div>
      </div>
    </footer>
  );
}
