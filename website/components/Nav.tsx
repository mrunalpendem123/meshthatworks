const REPO = 'https://github.com/mrunalpendem123/meshthatworks';

export function Nav() {
  return (
    <nav className="border-b border-line">
      <div className="max-w-page mx-auto px-6 h-14 flex items-center justify-between">
        <a href="/" className="text-fg font-bold tracking-wider !border-b-0">
          mesh<span className="text-fgDim">thatworks</span>
        </a>

        <div className="flex items-center gap-6 text-sm">
          <a href="#thesis" className="text-fgDim hover:text-fg !border-b-0">
            Thesis
          </a>
          <a href="#how" className="text-fgDim hover:text-fg !border-b-0">
            How
          </a>
          <a href="#status" className="text-fgDim hover:text-fg !border-b-0">
            Status
          </a>
          <a href={REPO} className="text-fgDim hover:text-fg !border-b-0">
            GitHub
          </a>
          <a href={REPO} className="btn">
            ↓ Get it
          </a>
        </div>
      </div>
    </nav>
  );
}
