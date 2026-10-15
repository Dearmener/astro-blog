import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import satori from 'satori';
import { html as satoriHtml } from 'satori-html';
import { Resvg } from '@resvg/resvg-js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const CACHE_DIR = join(ROOT, '.cache');
const FONTS_DIR = join(CACHE_DIR, 'fonts');
// Default output dir: dist/og (written after the build so it ships with the
// deployed static assets without polluting public/). The integration passes
// an override when invoked.
const DEFAULT_OUT_DIR = join(ROOT, 'dist', 'og');

/* ------------------------------------------------------------------
   Fonts

   Serif for titles, sans for everything else -- the same split the site
   uses. Both are CJK faces because most titles are Chinese; a Latin-only
   serif would leave the Han glyphs falling back to the sans face and the
   heading would read inconsistently within one line.
------------------------------------------------------------------ */

const FONTS = [
  {
    name: 'Noto Serif',
    url: 'https://github.com/googlefonts/noto-cjk/raw/main/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf',
    file: 'NotoSerifSC-Regular.otf',
  },
  {
    name: 'Noto Sans',
    url: 'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf',
    file: 'NotoSansSC-Regular.otf',
  },
];

async function loadFonts() {
  mkdirSync(FONTS_DIR, { recursive: true });
  const out = [];
  for (const f of FONTS) {
    const path = join(FONTS_DIR, f.file);
    if (!existsSync(path) || readFileSync(path).length < 1_000_000) {
      console.log(`[og] downloading ${f.name} (one-time, ~16MB)...`);
      const res = await fetch(f.url, { redirect: 'follow' });
      if (!res.ok) throw new Error(`Failed to download ${f.name}: ${res.status}`);
      writeFileSync(path, Buffer.from(await res.arrayBuffer()));
      console.log(`[og] ${f.name} cached at ${path}`);
    }
    out.push({
      name: f.name,
      data: readFileSync(path),
      weight: 400,
      style: 'normal',
    });
  }
  return out;
}

/* ------------------------------------------------------------------
   Palette

   Mirrors the `.dark` block in src/styles/global.css. OG cards are always
   dark: they are seen as small thumbnails in a feed, where a light card
   washes out, and dark also matches the code blocks readers associate
   with the site. hue is supplied per post by the integration so a series
   card matches its rows in list views.
------------------------------------------------------------------ */

const COLORS = {
  bg: 'hsl(174 23% 7%)',
  text: 'hsl(150 12% 92%)',
  textDim: 'hsl(170 10% 62%)',
  textMuted: 'hsl(170 8% 46%)',
  line: 'hsl(172 14% 19%)',
  accent: 'hsl(124 70% 55%)',
};

function escapeXml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function truncate(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function buildTemplate({ title, description, category, tags, author, siteName, isArticle, hue = 173 }) {
  const h = Number(hue);

  // Outlined pills rather than filled ones: at thumbnail size a filled chip
  // reads as a blob, an outline still reads as a tag.
  const tagBadges = (tags || [])
    .slice(0, 3)
    .map(
      (t) =>
        `<div style="display:flex;padding:6px 14px;border-radius:999px;border:1px solid hsl(${h} 30% 32%);color:hsl(${h} 65% 72%);font-size:21px;">${escapeXml(t)}</div>`,
    )
    .join('');

  // One hairline accent bar, echoing the rule the site draws beside every
  // section heading.
  const accentBar = `<div style="display:flex;width:56px;height:4px;border-radius:2px;background:hsl(${h} 48% 52%);margin-bottom:22px;"></div>`;

  const categoryChip =
    isArticle && category
      ? `<div style="display:flex;align-items:center;gap:8px;padding:7px 16px;border-radius:3px;border:1px solid hsl(${h} 30% 30%);color:hsl(${h} 70% 74%);font-size:20px;font-weight:500;"><div style="display:flex;width:7px;height:7px;border-radius:50%;background:hsl(${h} 70% 21%);"></div>${escapeXml(category)}</div>`
      : '';

  return `<div style="display:flex;flex-direction:column;width:1200px;height:630px;background:${COLORS.bg};font-family:'Noto Sans';position:relative;">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;background:hsl(${h} 70% 21%);display:flex;"></div>

  <div style="display:flex;flex-direction:column;flex:1;padding:56px 72px;position:relative;">

    <div style="display:flex;align-items:center;justify-content:space-between;">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="display:flex;align-items:center;justify-content:center;width:52px;height:52px;border-radius:3px;background:hsl(${h} 70% 21%);color:#fff;font-family:'Noto Serif';font-size:26px;font-weight:700;">孟</div>
        <div style="display:flex;flex-direction:column;">
          <div style="color:${COLORS.text};font-size:23px;font-weight:600;line-height:1.25;">${escapeXml(siteName)}</div>
          <div style="color:${COLORS.textMuted};font-size:17px;margin-top:2px;">menggq.pages.dev</div>
        </div>
      </div>
      ${categoryChip}
    </div>

    <div style="display:flex;flex-direction:column;margin-top:auto;margin-bottom:auto;max-width:1010px;">${accentBar}<div style="font-family:'Noto Serif';color:${COLORS.text};font-size:${isArticle ? 58 : 66}px;font-weight:700;line-height:1.28;letter-spacing:-0.01em;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;">${escapeXml(truncate(title, isArticle ? 44 : 20))}</div>
      ${
        description && isArticle
          ? `<div style="color:${COLORS.textDim};font-size:25px;line-height:1.6;margin-top:22px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;max-width:960px;">${escapeXml(truncate(description, 110))}</div>`
          : ''
      }
    </div>

    <div style="display:flex;align-items:center;justify-content:space-between;">
      <div style="display:flex;gap:10px;flex-wrap:wrap;">${tagBadges}</div>
      <div style="display:flex;align-items:center;gap:10px;color:${COLORS.textMuted};font-size:21px;">${escapeXml(author || '孟国庆')}</div>
    </div>
  </div>
</div>`;
}

async function renderOgPng({ fonts, template }) {
  const tree = satoriHtml(template);
  const svg = await satori(tree, {
    width: 1200,
    height: 630,
    fonts,
  });
  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: 1200 },
    font: {
      loadSystemFonts: false,
    },
  });
  return resvg.render().asPng();
}

function safeSlug(s) {
  return String(s)
    .replace(/[\/\\]/g, '-')
    .replace(/[^a-zA-Z0-9\u4e00-\u9fa5\-_]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 120);
}

async function main() {
  const fonts = await loadFonts();

  const arg = process.argv[2];
  if (!arg) {
    console.error('[og] usage: node scripts/generate-og.mjs <jobs-json> [out-dir]');
    process.exit(1);
  }
  const jobs = JSON.parse(arg);
  const outDir = process.argv[3] ? resolve(process.argv[3]) : DEFAULT_OUT_DIR;
  mkdirSync(outDir, { recursive: true });

  const siteName = jobs.siteName || '孟国庆的博客';
  let count = 0;
  for (const job of jobs.posts) {
    const tpl = buildTemplate({
      title: job.title,
      description: job.description,
      category: job.category,
      tags: job.tags || [],
      author: job.author || '孟国庆',
      siteName,
      hue: job.hue,
      isArticle: job.isArticle !== false,
    });
    const png = await renderOgPng({ fonts, template: tpl });
    const file = safeSlug(job.slug) + '.png';
    writeFileSync(join(outDir, file), png);
    count++;
  }
  console.log(`[og] generated ${count} og images -> ${outDir}`);
}

main().catch((err) => {
  console.error('[og] error:', err);
  process.exit(1);
});