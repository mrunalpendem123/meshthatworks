import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0a0a0a',
        bgElev: '#111111',
        fg: '#d4d4d4',
        fgDim: '#8a8a8a',
        muted: '#5c5c5c',
        line: '#222222',
        accent: '#e8e8e8',
        ok: '#67d067',
        warn: '#d8b46a',
        err: '#d86a6a',
      },
      fontFamily: {
        mono: ['var(--font-mono)', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      maxWidth: {
        page: '960px',
      },
    },
  },
  plugins: [],
};

export default config;
