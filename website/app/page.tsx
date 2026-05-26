import { Nav } from '@/components/Nav';
import { Footer } from '@/components/Footer';
import { Globe } from '@/components/Globe';
import { Reveal } from '@/components/Reveal';
import { CopyableCommand } from '@/components/CopyableCommand';
import { MeshMark } from '@/components/Brand';

const REPO = 'https://github.com/mrunalpendem123/meshthatworks';
const RELEASES = `${REPO}/releases/latest`;
// Direct-download URL: GitHub 302s straight to the asset with a download header.
// The split-capable build, shipped as a .zip (unzip → right-click Open, or
// `xattr -dr com.apple.quarantine` for the un-notarized app).
const DMG = `${REPO}/releases/latest/download/MeshThatWorks.zip`;
const INSTALL_CMD =
  'curl -fsSL https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh | sh';
// Run once after downloading — the app isn't App-Store-notarized, so this clears
// the quarantine flag and opens it.
const OPEN_CMD =
  'cd ~/Downloads && unzip -o MeshThatWorks.zip && xattr -dr com.apple.quarantine MeshThatWorks.app && open MeshThatWorks.app';

export default function Page() {
  return (
    <>
      <Nav />

      {/* ───────────────────────────── hero */}
      <header className="relative overflow-hidden">
        {/* globe drifting up behind the hero */}
        <div className="pointer-events-none absolute left-1/2 top-[46%] -translate-x-1/2 -translate-y-1/2 opacity-50 sm:opacity-90">
          <Globe size={680} />
        </div>

        <div className="relative mx-auto flex max-w-page flex-col items-center px-6 pb-24 pt-20 text-center">
          <div className="pill animate-fade-up px-3.5 py-1.5 text-[12.5px] text-ink-1">
            <span className="h-2 w-2 animate-breathe rounded-full bg-node" style={{ boxShadow: '0 0 8px #5ce8a6' }} />
            Now a native macOS app
          </div>

          <h1 className="animate-fade-up mt-7 max-w-3xl text-balance text-4xl font-semibold leading-[1.1] tracking-tight text-ink-0 md:text-6xl">
            Frontier AI on the Macs you already own.
          </h1>

          <p className="animate-fade-up mt-5 max-w-xl text-[15px] leading-relaxed text-ink-2 md:text-base">
            The biggest open models want 16&nbsp;GB of RAM per device. Most Macs have 8.
            MeshThatWorks treats your SSD as memory and splits models across paired
            devices — so a model that asks for 18&nbsp;GB runs on a Mac with 8.
          </p>

          <div className="animate-fade-up mt-9 flex flex-wrap items-center justify-center gap-3">
            <a href={DMG} className="pill pill-primary px-6 py-3 text-[15px] no-underline">
              ↓ Download for Mac
            </a>
            <a href={REPO} className="pill px-5 py-3 text-[14px] text-ink-0 no-underline">
              View on GitHub
            </a>
          </div>
          <p className="animate-fade-up mt-3 text-[12px] text-ink-3">
            Apple Silicon · macOS 12+ · signed &amp; notarized · MIT
          </p>
        </div>
      </header>

      {/* ───────────────────────────── pillars */}
      <section className="mx-auto max-w-page px-6 py-10">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {[
            {
              title: 'Your SSD becomes memory',
              body: 'MoE models fire only a few experts per token. Weights stream from disk on demand, so the working set stays small even when the model is huge.',
            },
            {
              title: 'Your devices share the work',
              body: 'Pair two Macs and the model splits across them — each holds a slice of the layers. Activations cross the wire over QUIC, a few MB per token.',
            },
            {
              title: 'Private by construction',
              body: 'No cloud, no accounts, no telemetry. End-to-end encrypted over iroh, addressed by keys you generate locally. Nothing leaves your hardware.',
            },
          ].map((p, i) => (
            <Reveal key={p.title} delay={i * 0.08}>
              <div className="glass h-full rounded-2xl p-6">
                <MeshMark size={22} />
                <h3 className="mt-4 text-[16px] font-semibold text-ink-0">{p.title}</h3>
                <p className="mt-2 text-[13.5px] leading-relaxed text-ink-2">{p.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ───────────────────────────── how it works */}
      <section id="how" className="mx-auto max-w-page px-6 py-16">
        <Reveal>
          <h2 className="text-2xl font-semibold tracking-tight text-ink-0">How it works</h2>
          <p className="mt-2 max-w-2xl text-[14px] text-ink-2">
            One forward pass crossing two paired devices. The orchestrator is whichever
            Mac you typed at; the other is a layer-forward target.
          </p>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="glass mt-6 overflow-x-auto rounded-2xl p-6">
            <pre className="mono text-[12px] leading-relaxed text-ink-1">
{`  ┌──────────── device A (8 GB Mac) ─────────────┐
  │   SSD ─→ page cache ─→ MLX (Metal) on layers │
  │                            0..K              │
  └──────────────────────────────────────────────┘
                       │  activation [batch, seq, hidden]
                       ▼  ~2 MB · QUIC over iroh
  ┌──────────── device B (8 GB Mac) ─────────────┐
  │   SSD ─→ page cache ─→ MLX (Metal) on layers │
  │                          K..N                │
  └──────────────────────────────────────────────┘
                       │  logits
                       ▼  → next token`}
            </pre>
          </div>
        </Reveal>
      </section>

      {/* ───────────────────────────── thesis */}
      <section id="thesis" className="mx-auto max-w-page px-6 py-16">
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1fr_360px] lg:items-center">
          <Reveal>
            <h2 className="text-2xl font-semibold tracking-tight text-ink-0">The thesis</h2>
            <div className="mt-4 space-y-4 text-[14px] leading-relaxed text-ink-2">
              <p>
                <span className="text-ink-0">Compute is all around us.</span> Most people own
                two or three Apple devices that sit idle most of the day. Together they have
                plenty of memory and SSD bandwidth to run frontier models — they just can&apos;t,
                alone.
              </p>
              <p>
                SwiftLM proved you can stream weights from SSD on a single big Mac. Mesh-LLM and
                friends split models across devices — but assume each already has the RAM.{' '}
                <span className="text-ink-0">Nobody combined SSD streaming with mesh distribution.</span>{' '}
                That&apos;s the gap MeshThatWorks closes, dropping the per-device floor from 16&nbsp;GB
                to 4–8&nbsp;GB.
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.1} className="flex justify-center">
            <div className="animate-float">
              <Globe size={300} />
            </div>
          </Reveal>
        </div>
      </section>

      {/* ───────────────────────────── status */}
      <section className="mx-auto max-w-page px-6 py-16">
        <Reveal>
          <h2 className="text-2xl font-semibold tracking-tight text-ink-0">What runs today</h2>
        </Reveal>
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
          {[
            { stat: 'Streaming proven', detail: 'RSS oscillated 30 KB ↔ 906 MB on an 18 GB model on an 8 GB Mac.' },
            { stat: 'OpenAI-compatible', detail: 'Local proxy on :9337 — Claude Code, Cursor, the OpenAI SDK all just work.' },
            { stat: '4 ALPNs live', detail: 'health · infer · layer · layer-forward, over NAT-traversed QUIC.' },
          ].map((s, i) => (
            <Reveal key={s.stat} delay={i * 0.08}>
              <div className="glass h-full rounded-2xl p-6">
                <div className="text-[15px] font-semibold text-node">{s.stat}</div>
                <p className="mt-2 text-[13px] leading-relaxed text-ink-2">{s.detail}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* ───────────────────────────── get started */}
      <section className="mx-auto max-w-page px-6 py-16">
        <Reveal>
          <div className="glass rounded-3xl p-8 md:p-10">
            <h2 className="text-2xl font-semibold tracking-tight text-ink-0">Get started</h2>
            <p className="mt-2 max-w-2xl text-[14px] text-ink-2">
              Two ways in. The Mac app walks you through everything; the CLI is for terminal folks.
            </p>

            <div className="mt-7 grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="rounded-2xl border border-line bg-white/[0.02] p-6">
                <div className="text-[13px] font-medium uppercase tracking-wider text-ink-3">Recommended</div>
                <h3 className="mt-1 text-[17px] font-semibold text-ink-0">The Mac app</h3>
                <p className="mt-2 text-[13px] text-ink-2">
                  Download, open, and let the onboarding set up the engine and a model. Globe,
                  chat, and mesh pairing built in.
                </p>
                <a href={DMG} className="pill pill-primary mt-4 px-5 py-2.5 text-[14px] no-underline">
                  ↓ Download for Mac
                </a>
                <p className="mt-4 text-[12px] text-ink-3">Then run this once to open it:</p>
                <div className="mt-2 rounded-xl border border-line bg-black/40 p-3">
                  <CopyableCommand cmd={OPEN_CMD} />
                </div>
              </div>

              <div className="rounded-2xl border border-line bg-white/[0.02] p-6">
                <div className="text-[13px] font-medium uppercase tracking-wider text-ink-3">For terminals</div>
                <h3 className="mt-1 text-[17px] font-semibold text-ink-0">One-line install</h3>
                <p className="mt-2 text-[13px] text-ink-2">
                  Installs the <code className="text-node">mtw</code> CLI + dashboard. First run
                  builds SwiftLM (~30 min).
                </p>
                <div className="mt-4 rounded-xl border border-line bg-black/40 p-3">
                  <CopyableCommand cmd={INSTALL_CMD} />
                </div>
              </div>
            </div>

            <p className="mt-5 text-[12.5px] text-ink-3">
              Point any OpenAI-compatible app at{' '}
              <code className="text-ink-1">http://localhost:9337</code> once it&apos;s running.
            </p>
          </div>
        </Reveal>
      </section>

      <Footer />
    </>
  );
}
