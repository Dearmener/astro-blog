/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        // Latin leads each stack so English words and digits are drawn by a
        // proper face; CJK falls through to the platform's system font.
        sans: [
          'system-ui',
          '-apple-system',
          '"PingFang SC"',
          '"Hiragino Sans GB"',
          '"Microsoft YaHei"',
          '"Noto Sans SC"',
          'sans-serif',
        ],
        serif: [
          'Charter',
          '"Bitstream Charter"',
          'Georgia',
          '"Songti SC"',
          '"Source Han Serif SC"',
          '"Noto Serif SC"',
          'STSong',
          'SimSun',
          'serif',
        ],
        mono: [
          '"Paper Mono"',
          'ui-monospace',
          'SFMono-Regular',
          '"SF Mono"',
          'Menlo',
          'Consolas',
          '"Liberation Mono"',
          '"Courier New"',
          'monospace',
        ],
      },
      colors: {
        // Aliases onto the HSL channel tokens defined in src/styles/global.css.
        fg: 'hsl(var(--fg))',
        muted: 'hsl(var(--fg-muted))',
        faint: 'hsl(var(--fg-faint))',
        surface: 'hsl(var(--bg))',
        subtle: 'hsl(var(--bg-subtle))',
        hover: 'hsl(var(--bg-hover))',
        line: 'hsl(var(--line))',
        'line-strong': 'hsl(var(--line-strong))',
        brand: 'hsl(var(--brand))',
      },
      borderRadius: {
        DEFAULT: '0.25rem',
      },
    },
  },
  plugins: [],
};