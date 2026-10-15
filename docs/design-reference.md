# Design reference — zendot.org

Traversed 2026-10-09. Pages sampled: `/en`, `/en/rankings`, `/en/domains/misc`,
`/en/domains/dev-ops`, `/en/domains/gen-ai`, `/en/posts/autumn-27-artex`,
`/en/submit`, `/en/signals`.

All eight pages ship **one** stylesheet (`/_next/static/css/86e397070a2fc73d.css`,
32 KB, Tailwind 3.4.19). The design system is therefore *tokens + inline utility
classes*, not a component library: only four purpose-built classes exist.

## Tokens

Light theme (HSL channel triples, so `hsl(var(--x) / .4)` works):

| token | value |
|---|---|
| `--background` | `150 20% 98%` |
| `--foreground` | `174 30% 12%` |
| `--card` | `0 0% 100%` |
| `--primary` | `173 58% 27%` |
| `--secondary` | `158 22% 94%` |
| `--muted` | `156 16% 95%` |
| `--muted-foreground` | `172 12% 38%` |
| `--accent` | `156 26% 91%` |
| `--accent-foreground` | `174 30% 14%` |
| `--border` / `--input` | `158 16% 85%` |
| `--ring` | `173 58% 30%` |
| `--radius` | `0.25rem` |
| `--domain-ink` | `199 72% 30%` (per-category override) |
| `--breaking` | `12 74% 46%` |

Domain accent colours are per-category, injected inline
(`style="background-color:hsl(188 55% 30%)"`).

## The four custom classes

```css
.cn-article { font-size: 1.0625rem; line-height: 1.9; letter-spacing: .01em }
.cn-num     { font-family: JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
              font-variant-numeric: tabular-nums; letter-spacing: -.01em }
.cn-qing    { background-image: linear-gradient(110deg,
                hsl(199 62% 27%), hsl(184 55% 27%) 45%, hsl(157 46% 30%)) }
.cn-rule    { border-top-width: 1px; border-color: hsl(var(--foreground) / .15) }
.max-w-page { max-width: 1215px }
```

`cn-qing` (青) is the brand gradient — a **three-stop** teal→green sweep. It
backs the whole nav bar, not just the logo.

## Font stacks

```
sans   PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans SC, system-ui, sans-serif
serif  Songti SC, "Source Han Serif SC", "Noto Serif SC", STSong, SimSun, serif
mono   JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace
```

Serif is used for **page titles and article section headings only** — never for
list-row titles, nav, or body copy.

## Page shell

```html
<main class="mx-auto max-w-page px-5 py-8 pb-10">
  <div class="grid gap-10 lg:grid-cols-[minmax(0,1fr)_17rem]">
    <section class="min-w-0"> … content … </section>
    <aside class="hidden lg:block"> … 17rem sidebar … </aside>
```

`max-w-page` is 1215px with `px-5`. Content column plus a 17rem rail.

## Header — two stacked bars

1. **Utility bar** — `border-b border-border/70 bg-card`, inner
   `mx-auto flex max-w-page flex-wrap items-center justify-between gap-x-6 gap-y-2 px-5 py-3`.
   Wordmark `font-sans text-[22px] font-semibold leading-none tracking-tight md:text-[26px]`,
   tagline `hidden text-xs text-muted-foreground lg:inline`.
   Search `h-8 w-36 rounded-sm border border-border bg-background px-2.5 text-xs … md:w-44`.
2. **Nav bar** — `cn-qing text-white`, inner
   `mx-auto flex max-w-page flex-wrap items-center gap-x-5 px-5`.
   Links `border-b-2 border-transparent py-2 text-[13px] text-white/85 hover:text-white`;
   active adds `border-white/70 font-semibold text-white`.

## Page heading block

```html
<nav class="flex items-center text-xs text-muted-foreground">
  <a class="hover:text-brand">Home</a><span class="mx-1.5">/</span><span>Category</span>
  <a class="ml-auto text-brand hover:underline">RSS</a>
</nav>
<div class="mt-4">
  <div class="flex flex-wrap items-center gap-x-3 gap-y-3 border-b-2 border-primary pb-4">
    <div class="flex items-center gap-3">
      <span class="inline-block h-3 w-3 shrink-0 rounded-sm" style="background-color:hsl(188 55% 30%)"></span>
      <h1 class="font-serif text-2xl font-bold leading-tight md:text-3xl">…</h1>
    </div>
    <p class="text-xs text-muted-foreground">196 picks in this category, newest first</p>
```

A 12px rounded colour chip precedes the serif title; the block is closed by a
**2px** primary rule, not a 1px border.

## List row

```html
<article class="group relative border-b border-dashed border-border py-5">
  <header class="relative rounded-sm py-2 pl-3 pr-[4.5rem]"
          style="background:linear-gradient(120deg, hsl(var(--primary)/0.96), hsl(var(--domain-ink)))">
    <h2 class="min-w-0 text-[17px] font-semibold leading-snug text-white md:text-lg">
      <a class="hover:underline">…</a></h2>
    <!-- optional logo: h-11 w-11 rounded-sm absolute -bottom-3 right-3 object-contain -->
  </header>
  <div class="cn-num mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-sm bg-secondary/60
              py-1.5 pl-3 text-xs text-muted-foreground pr-3">
    <span class="font-sans">Published</span><time>…</time><span aria-hidden="true">·</span>…
  </div>
  <p class="mt-2.5 line-clamp-2 px-3 text-sm leading-6 text-muted-foreground">…</p>
  <div class="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 px-3 text-xs">
    <a class="font-medium text-brand hover:underline">Read the full pick →</a>
    <a class="text-muted-foreground hover:text-brand">Repo ↗</a>
    <a class="rounded-sm bg-accent px-1.5 py-0.5 text-accent-foreground
              transition-colors hover:bg-primary hover:text-primary-foreground">Category</a>
  </div>
</article>
```

Notes that matter:

- The band title is **sans**, `font-semibold` — the serif face is reserved for
  `h1` and article `h2`.
- The meta strip's `pr-3` only becomes `pr-[4.5rem]` when a logo mark exists.
- The mark is `-bottom-3 right-3`, i.e. it straddles the band's lower edge, and
  uses `object-contain` on a transparent plate — it works because project logos
  are square, self-contained marks. Rows without a logo simply omit it.
- Rows are separated by **dashed** rules; the only solid rules on the page are
  the 2px heading underline and `.cn-rule`.

## Article page

```html
<div class="mt-4 flex flex-wrap items-center gap-2">   <!-- tag pills -->
  <a class="inline-flex shrink-0 items-center rounded-sm px-2 py-1.5 text-[12px] font-medium
            leading-none transition-colors"
     style="color:hsl(24 62% 30%);background-color:hsl(24 62% 94%)">AI Agent</a>
</div>
<h1 class="mt-4 font-serif text-[28px] font-bold leading-snug md:text-[36px] md:leading-[1.3]">
<div class="cn-rule mt-5 flex flex-wrap items-center gap-x-4 gap-y-1 pt-3 text-[13px] text-muted-foreground">
```

The tag pills are the one place colour is allowed to vary freely — and the
mechanism is the key detail: **hue is derived per tag while saturation and
lightness are pinned** (`62%` / `30%` and `94%`). Arbitrary per-item colour
rotation (index-based palettes) is what makes a tag cloud look cheap; a
content-derived hue with fixed chroma does not.

Prose, inside `.cn-article`:

| element | styling |
|---|---|
| `h2` | `mt-12 mb-4 scroll-mt-6 border-l-[3px] border-brand pl-3 font-serif text-[22px] font-bold leading-snug` |
| `p` | inherited — 17px / 1.9 / .01em |
| `ul` | `my-5 list-disc space-y-2 pl-6 marker:text-muted-foreground` |
| `li` | `pl-1` |
| `a` | `text-brand underline decoration-brand/40 underline-offset-2 hover:decoration-brand` |

Sidebar rail:

```html
<aside class="hidden lg:block">
  <div class="sticky top-8 space-y-9">
    <nav aria-label="On this page">
      <p class="text-xs font-semibold tracking-[0.12em] text-muted-foreground">On this page</p>
      <ul class="mt-3 space-y-2 border-l border-border pl-4">
        <li><a class="text-[13px] leading-6 text-muted-foreground transition-colors hover:text-brand">…</a></li>
```

Section labels across the site use `text-xs font-semibold tracking-[0.12em]
text-muted-foreground`; homepage section headers use the looser
`text-sm font-semibold tracking-wide text-muted-foreground` under a 2px primary rule.

## Footer

```html
<footer class="border-t border-border/70 bg-secondary/40">
  <div class="mx-auto flex max-w-page flex-col gap-2 px-5 py-10 text-xs text-muted-foreground
              md:flex-row md:items-center md:justify-between">
```

## Contrast

Measured on the light palette: `foreground`/`background` 14.8:1,
`muted-foreground`/`background` 5.35:1, `primary`/`background` 5.9:1,
`primary`/`secondary` 5.5:1. `border` on `background` is only 1.3:1 — borders
are deliberately hairlines and never carry meaning on their own.

## Conclusions to apply

1. Page width is 1215px, not the ~800px the current blog uses.
2. Nav is a brand-gradient band, not a white bar with a 1px border.
3. List-row band titles are sans; serif is for `h1` and article `h2` only.
4. Heading blocks close with a 2px primary rule.
5. Tag colour is allowed, but derived per tag with pinned saturation/lightness.
6. Row meta strips are flush right unless a mark is present.