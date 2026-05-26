import { Wordmark } from './Brand';

const REPO = 'https://github.com/mrunalpendem123/meshthatworks';
const DMG = `${REPO}/releases/latest/download/MeshThatWorks.zip`;

export function Nav() {
  return (
    <nav className="sticky top-0 z-50 px-6 pt-4">
      <div className="glass mx-auto flex h-14 max-w-page items-center justify-between rounded-2xl px-5">
        <a href="/" className="no-underline">
          <Wordmark />
        </a>
        <div className="flex items-center gap-1 text-[13px]">
          <a href="#how" className="hidden rounded-full px-3.5 py-1.5 text-ink-2 transition-colors hover:text-ink-0 sm:block">
            How it works
          </a>
          <a href="#thesis" className="hidden rounded-full px-3.5 py-1.5 text-ink-2 transition-colors hover:text-ink-0 sm:block">
            Thesis
          </a>
          <a href={REPO} className="hidden rounded-full px-3.5 py-1.5 text-ink-2 transition-colors hover:text-ink-0 sm:block">
            GitHub
          </a>
          <a href={DMG} className="pill pill-primary ml-1 px-4 py-1.5 text-[13px] no-underline">
            Download
          </a>
        </div>
      </div>
    </nav>
  );
}
