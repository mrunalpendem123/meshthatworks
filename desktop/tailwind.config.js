/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Deep-space palette inspired by the LayerEdge reference.
        space: {
          0: '#05060d', // base void
          1: '#0a0c16', // panel
          2: '#10131f', // elevated panel
          3: '#171b2b', // hover / border-elevated
        },
        ink: {
          0: '#f4f6fb', // primary text
          1: '#b9c0d4', // secondary
          2: '#7c849b', // tertiary
          3: '#4d5469', // muted
        },
        line: 'rgba(255,255,255,0.08)',
        node: {
          // signature mesh-green (the radar dot in the reference)
          DEFAULT: '#5ce8a6',
          dim: '#2f9d70',
          glow: 'rgba(92,232,166,0.45)',
        },
        aurora: {
          violet: '#7b61ff',
          amber: '#ff9f6b',
          teal: '#46d6c8',
        },
      },
      fontFamily: {
        sans: ['SF Pro Display', '-apple-system', 'BlinkMacSystemFont', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['SF Mono', 'ui-monospace', 'JetBrains Mono', 'Menlo', 'monospace'],
      },
      boxShadow: {
        glass: 'inset 0 1px 0 0 rgba(255,255,255,0.06), 0 8px 30px -10px rgba(0,0,0,0.7)',
        node: '0 0 0 1px rgba(92,232,166,0.25), 0 0 24px -2px rgba(92,232,166,0.35)',
      },
      backdropBlur: { xs: '2px' },
      keyframes: {
        'spin-slow': { to: { transform: 'rotate(360deg)' } },
        ripple: {
          '0%': { transform: 'scale(0.6)', opacity: '0.7' },
          '100%': { transform: 'scale(2.2)', opacity: '0' },
        },
        breathe: {
          '0%,100%': { opacity: '0.5' },
          '50%': { opacity: '1' },
        },
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      animation: {
        'spin-slow': 'spin-slow 8s linear infinite',
        ripple: 'ripple 3s ease-out infinite',
        breathe: 'breathe 2.4s ease-in-out infinite',
        'fade-up': 'fade-up 0.5s cubic-bezier(0.16,1,0.3,1) both',
        shimmer: 'shimmer 2.5s linear infinite',
      },
    },
  },
  plugins: [],
};
