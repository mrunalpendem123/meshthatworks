import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        space: { 0: '#05060d', 1: '#0a0c16', 2: '#10131f', 3: '#171b2b' },
        ink: { 0: '#f4f6fb', 1: '#b9c0d4', 2: '#7c849b', 3: '#4d5469' },
        line: 'rgba(255,255,255,0.08)',
        node: { DEFAULT: '#5ce8a6', dim: '#2f9d70' },
        aurora: { violet: '#7b61ff', amber: '#ff9f6b', teal: '#46d6c8' },
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      maxWidth: { page: '1080px' },
      keyframes: {
        ripple: {
          '0%': { transform: 'scale(0.6)', opacity: '0.7' },
          '100%': { transform: 'scale(2.2)', opacity: '0' },
        },
        breathe: { '0%,100%': { opacity: '0.5' }, '50%': { opacity: '1' } },
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(14px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        float: {
          '0%,100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-8px)' },
        },
      },
      animation: {
        ripple: 'ripple 3s ease-out infinite',
        breathe: 'breathe 2.4s ease-in-out infinite',
        'fade-up': 'fade-up 0.7s cubic-bezier(0.16,1,0.3,1) both',
        float: 'float 6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};

export default config;
