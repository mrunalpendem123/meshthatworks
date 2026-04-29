# meshthatworks website

Marketing + thesis page for MeshThatWorks. Single-page Next.js 15 + Tailwind, monospace
opencode-style aesthetic.

## Local dev

```
npm install
npm run dev
```

Then open http://localhost:3000.

## Deploy to Vercel

```
npm install
npx vercel deploy --prod
```

The first time, the Vercel CLI prompts for login + project name. After that,
subsequent deploys are one command.

## What lives here

- `app/page.tsx` — the whole page in sections (hero, what is, thesis, how it
  works, status, privacy, quickstart, footer).
- `components/` — small page-level pieces (Nav, Section, Bullet, Footer).
- `next.config.mjs` — rewrites `/install` to the bootstrap script on GitHub
  so `curl -fsSL <site>/install | sh` works.

## Editing

The thesis text is in `app/page.tsx` under the "thesis" section. The crate
descriptions and ALPN list mirror the main repo's README — keep them in sync
when the architecture changes.
