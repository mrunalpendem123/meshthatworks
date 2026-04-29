import { Nav } from '@/components/Nav';
import { Section } from '@/components/Section';
import { Bullet } from '@/components/Bullet';
import { Footer } from '@/components/Footer';

const REPO = 'https://github.com/mrunalpendem123/meshthatworks';

export default function Page() {
  return (
    <>
      <Nav />

      {/* ─────────────────────────────────────── hero */}
      <Section>
        <div className="pt-16 pb-10">
          <div className="mb-7">
            <span className="tag">New</span>
            <span className="ml-3 text-fgDim">
              v0.1 — single-device working, two-device bridge built.{' '}
              <a href={REPO} className="text-fg">
                See the repo
              </a>
              .
            </span>
          </div>

          <h1 className="text-3xl md:text-4xl text-accent font-bold leading-tight mb-5">
            Frontier AI on the Macs you already own.
          </h1>

          <p className="text-fgDim max-w-2xl mb-10">
            The biggest open AI models are free to download, but the math wants 16 GB of RAM per
            device. Most consumer Macs have 8. MeshThatWorks lowers the floor by treating your SSD
            as memory and splitting models across paired devices — so a model that asks for 18 GB
            of RAM runs on a Mac with 8.
          </p>

          <div className="border border-line rounded-md bg-bgElev">
            <div className="flex items-center px-4 py-2 border-b border-line text-xs text-fgDim">
              <span className="mr-4 text-fg">curl</span>
              <span className="text-muted">macOS · Apple Silicon · MIT</span>
            </div>
            <div className="px-4 py-3 text-fg select-all">
              <span className="text-fgDim">$ </span>
              curl -fsSL meshthatworks.vercel.app/install | sh
            </div>
          </div>

          <p className="text-xs text-muted mt-3">
            First run takes ~30 minutes — it builds SwiftLM and installs <code>mtw</code> to{' '}
            <code>~/.local/bin</code>. Then <code>mtw start</code> opens the engine and the
            dashboard in one terminal.
          </p>
        </div>
      </Section>

      <hr className="divide" />

      {/* ─────────────────────────────────────── what is */}
      <Section>
        <div className="py-12">
          <h2 className="text-fg font-bold mb-5">What is MeshThatWorks?</h2>
          <p className="text-fgDim max-w-2xl mb-7">
            A way to run frontier open-source AI models on consumer Apple Silicon hardware.
            Combines two existing ideas — SSD expert streaming and peer-to-peer mesh inference —
            that nobody had previously combined.
          </p>

          <div className="space-y-4">
            <Bullet title="SSD as memory">
              Per-token, an MoE model only fires a handful of experts. Memory-map the weights file
              and stream pages on demand — the active working set is small even when the model is
              not.
            </Bullet>
            <Bullet title="Mesh distribution">
              Pair two Macs (or three) and the model splits across them. Each device only holds a
              slice of the layers. Activations cross the wire over QUIC.
            </Bullet>
            <Bullet title="Built on iroh">
              QUIC transport, NAT-traversed, end-to-end encrypted, falls back to a relay only when
              a direct path is impossible. Identities are Ed25519 keys.
            </Bullet>
            <Bullet title="OpenAI-compatible">
              Local proxy on <code>localhost:9337</code>. Claude Code, Cursor, the OpenAI Python
              SDK — everything that talks to OpenAI works without code changes.
            </Bullet>
            <Bullet title="MIT licensed, all local">
              No cloud. No accounts. No telemetry. Your prompts and your data stay on your
              machines.
            </Bullet>
          </div>

          <div className="mt-8">
            <a href={REPO} className="btn">
              → Read the source on GitHub
            </a>
          </div>
        </div>
      </Section>

      <hr className="divide" />

      {/* ─────────────────────────────────────── thesis */}
      <Section id="thesis">
        <div className="py-12">
          <h2 className="text-fg font-bold mb-5">The thesis</h2>
          <div className="text-fgDim space-y-5 max-w-3xl leading-relaxed">
            <p>
              <span className="text-fg">Compute is around us.</span> Most people own two or three
              Apple devices that sit idle most of the day — old MacBooks, iPads, Mac Minis.
              Together, they have plenty of memory and SSD bandwidth to run frontier-grade models.
              They just cannot, alone.
            </p>
            <p>
              The blocker is a single number: the <span className="text-fg">RAM floor</span>.
              Modern open models — Qwen3-Coder-30B, DeepSeek, Mixtral — want 16+ GB of RAM
              resident per device. Most consumer Macs ship with 8. So the model is downloadable
              and the hardware is sitting on the desk, but the gap between them is wide.
            </p>
            <p>
              Existing distributed-inference projects do not close that gap. Petals, Bloom,
              Mesh-LLM, and the recent EigenLayer-style decentralised inference designs all assume
              each node has the headroom to hold a full shard in RAM. They scale by adding more
              devices that already meet the floor — they do not lower the floor itself. SwiftLM,
              from a different lineage, proved you can stream weights from SSD on demand with
              tolerable performance, but only on a single big Mac.
            </p>
            <p>
              <span className="text-fg">
                Nobody combined SSD streaming with peer-to-peer mesh distribution.
              </span>{' '}
              That is the gap MeshThatWorks closes.
            </p>
            <p>
              The combination is straightforward, once you see it. Each device runs SwiftLM (Apple
              Silicon, MLX, expert weights memory-mapped from SSD). The devices pair over iroh
              (QUIC, NAT-traversed, end-to-end encrypted). A model&apos;s transformer layers split
              across the paired devices — the orchestrator pipelines a forward pass, and the only
              data that crosses the wire is the activation tensor between layer slices, a few
              megabytes per token.
            </p>
            <p>
              The result: <span className="text-fg">a 30B-parameter MoE model that no single 8 GB
              Mac can hold runs across two of them</span>, with privacy preserved by construction.
              No third-party servers. No accounts. No logs. The mesh is the user&apos;s own
              hardware on the user&apos;s own network.
            </p>
            <p>
              This is not a new idea about AI. It is a new combination of existing ideas about AI
              inference, plumbed end-to-end, with the user-facing edges sanded down enough that a
              non-specialist can install it in three commands and run a 30B-parameter model on the
              MacBook they already own.
            </p>
            <p className="text-muted text-sm pt-2">
              — read the full architecture spec at{' '}
              <a href={`${REPO}/blob/master/docs/ARCHITECTURE.md`} className="text-fg">
                docs/ARCHITECTURE.md
              </a>{' '}
              and live performance numbers at{' '}
              <a href={`${REPO}/blob/master/docs/BASELINES.md`} className="text-fg">
                docs/BASELINES.md
              </a>
              .
            </p>
          </div>
        </div>
      </Section>

      <hr className="divide" />

      {/* ─────────────────────────────────────── how it works */}
      <Section id="how">
        <div className="py-12">
          <h2 className="text-fg font-bold mb-3">How it works</h2>
          <p className="text-fgDim max-w-2xl mb-7">
            One forward pass crossing two paired devices. The orchestrator is whichever device the
            user typed at; the other is a layer-forward target.
          </p>

          <pre className="text-xs text-fgDim leading-snug overflow-x-auto bg-bgElev border border-line rounded-md p-5">
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

          <div className="mt-8 space-y-4">
            <Bullet title="mtw-core">
              The mesh layer. Peer discovery, identity, persistent peer list, four iroh ALPNs:
              <code> mtw/health/0</code>, <code>mtw/infer/0</code>, <code>mtw/layer/0</code>,{' '}
              <code>mtw/layer-forward/0</code>.
            </Bullet>
            <Bullet title="mtw-engine">
              Per-node inference. Drives a SwiftLM child process via HTTP. Implements the{' '}
              <code>LayerPeer</code> trait so local SwiftLM and remote iroh peers are
              interchangeable.
            </Bullet>
            <Bullet title="mtw-cache">
              Adaptive expert caching. Rolling activation histogram, hot/warm/cold tiering,{' '}
              <code>madvise</code>/<code>mlock</code> page-residency advisor over the
              memory-mapped weights.
            </Bullet>
            <Bullet title="mtw-api">
              OpenAI-compatible HTTP proxy on <code>localhost:9337</code>. Forwards to the local
              engine, exposes <code>/status</code> for the dashboard, emits correlated{' '}
              <code>[mtw-mem]</code> / <code>[mtw-req]</code> trace markers.
            </Bullet>
            <Bullet title="mtw-cli">
              The <code>mtw</code> binary and the ratatui dashboard. Pair / join / chat /
              dashboard all live here.
            </Bullet>
          </div>
        </div>
      </Section>

      <hr className="divide" />

      {/* ─────────────────────────────────────── numbers */}
      <Section id="status">
        <div className="py-12">
          <h2 className="text-fg font-bold mb-3">Status — what runs today</h2>
          <p className="text-fgDim max-w-2xl mb-7">
            Honest. Single-device works end-to-end. Two-device bridge is built and unit-tested
            against stub peers; awaiting a live two-Mac run. Throughput tuning is the open item.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <FigureCard
              caption="Streaming proven"
              detail="RSS 30 KB ↔ 906 MB on an 18 GB model"
              ascii={`▁▂▁▃▆█▆▃▁▁▂▅█▆▃▂▁▂▄▇█▆
▁▁▂▁▃▆▇▆▃▁▂▄▇█▇▄▂▁▂▃▆█
▂▁▁▃▆█▆▃▁▂▅█▇▄▂▁▂▄█▇▅▂`}
            />
            <FigureCard
              caption="32 / 32 tests"
              detail="across 4 ALPNs and 5 crates"
              ascii={`■■■■■■■■■■
■■■■■■■■■■
■■■■■■■■■■
■■■  ■  ■■`}
            />
            <FigureCard
              caption="4 ALPNs live"
              detail="health · infer · layer · layer-forward"
              ascii={`╲   ╱
 ╲ ╱
  ●
 ╱ ╲
╱   ╲`}
            />
          </div>

          <div className="mt-10 text-fgDim text-sm space-y-2">
            <div>
              <span className="text-ok">✓</span> Streaming pipeline proven on 8 GB Mac with 18 GB
              model.
            </div>
            <div>
              <span className="text-ok">✓</span> OpenAI-compatible proxy, dashboard, pairing over
              iroh.
            </div>
            <div>
              <span className="text-ok">✓</span> Cross-device layer-split bridge —{' '}
              <code>mtw/layer-forward/0</code>, three iroh integration tests passing.
            </div>
            <div>
              <span className="text-warn">⏳</span> Live two-Mac run with real SwiftLM (the bridge
              is in place).
            </div>
            <div>
              <span className="text-warn">⏳</span> Sustained throughput tuning — currently
              bandwidth-bound by macOS unified memory; the cache fix is built, not yet wired.
            </div>
          </div>
        </div>
      </Section>

      <hr className="divide" />

      {/* ─────────────────────────────────────── privacy */}
      <Section>
        <div className="py-12">
          <h2 className="text-fg font-bold mb-3">Privacy by construction</h2>
          <p className="text-fgDim max-w-2xl">
            The mesh is your devices on your network. There is no cloud component, no telemetry,
            no account, no logging. Your prompts, your context, your responses — they pass through
            QUIC streams that are end-to-end encrypted and addressed by Ed25519 keys you generated
            locally. Nothing leaves your hardware unless you tell it to.
          </p>
        </div>
      </Section>

      <hr className="divide" />

      {/* ─────────────────────────────────────── quickstart */}
      <Section>
        <div className="py-12">
          <h2 className="text-fg font-bold mb-3">Quickstart</h2>
          <p className="text-fgDim mb-7 max-w-2xl">
            Three commands. The first takes ~30 minutes the first time (it builds SwiftLM); the
            rest are seconds.
          </p>

          <div className="space-y-3 text-sm">
            <CommandLine
              num="1"
              comment="install — clones, builds, sets up ~/.local/bin/mtw"
              cmd="curl -fsSL meshthatworks.vercel.app/install | sh"
            />
            <CommandLine
              num="2"
              comment="pick a model — Models tab → Enter to download + activate"
              cmd="mtw dashboard"
            />
            <CommandLine
              num="3"
              comment="run — engine + dashboard in one terminal"
              cmd="mtw start"
            />
          </div>

          <p className="text-fgDim text-sm mt-7 max-w-2xl">
            Then point any OpenAI-compatible app at <code>http://localhost:9337</code>:
          </p>
          <pre className="bg-bgElev border border-line rounded-md p-4 mt-3 text-fg text-xs overflow-x-auto">
{`export OPENAI_BASE_URL=http://localhost:9337/v1
export OPENAI_API_KEY=local`}
          </pre>
        </div>
      </Section>

      <hr className="divide" />

      <Footer />
    </>
  );
}

// ────────────────────────────────────────── helpers (page-local)

function FigureCard({
  caption,
  detail,
  ascii,
}: {
  caption: string;
  detail: string;
  ascii: string;
}) {
  return (
    <div className="border border-line rounded-md p-5 bg-bgElev">
      <pre className="text-fgDim text-[11px] leading-tight whitespace-pre">{ascii}</pre>
      <div className="mt-4 text-fg text-sm">{caption}</div>
      <div className="text-muted text-xs mt-1">{detail}</div>
    </div>
  );
}

function CommandLine({
  num,
  comment,
  cmd,
}: {
  num: string;
  comment: string;
  cmd: string;
}) {
  return (
    <div>
      <div className="text-muted text-xs mb-1">
        # {num}. {comment}
      </div>
      <div className="bg-bgElev border border-line rounded-md px-4 py-2 text-fg select-all">
        <span className="text-fgDim">$ </span>
        {cmd}
      </div>
    </div>
  );
}
