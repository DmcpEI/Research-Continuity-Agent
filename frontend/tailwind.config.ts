import type { Config } from 'tailwindcss';

export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0f',
        surface: '#111118',
        border: '#1e1e2e',
        'text-primary': '#e2e2e8',
        'text-secondary': '#6b6b7e',
        'text-mono': '#a0a0b8',
        accent: '#7c6af7',
        green: '#22c55e',
        amber: '#f59e0b',
      },
      fontFamily: {
        sans: ['IBM Plex Sans', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config;
