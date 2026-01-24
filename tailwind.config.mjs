/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: [
          'ui-sans-serif',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Helvetica',
          'Apple Color Emoji',
          'Arial',
          'sans-serif',
          'Segoe UI Emoji',
          'Segoe UI Symbol',
        ],
        mono: [
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          'PT Mono',
          'Liberation Mono',
          'Courier',
          'monospace',
        ],
      },
      colors: {
        notion: {
          text: 'var(--notion-text)',
          'text-secondary': 'var(--notion-text-secondary)',
          bg: 'var(--notion-bg)',
          'bg-secondary': 'var(--notion-bg-secondary)',
          'bg-hover': 'var(--notion-bg-hover)',
          border: 'var(--notion-border)',
          link: 'var(--notion-link)',
        },
      },
    },
  },
  plugins: [],
};
