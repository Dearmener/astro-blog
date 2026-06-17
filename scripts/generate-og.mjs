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

const FONT_URL =
  'https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf';
const FONT_PATH = join(FONTS_DIR, 'NotoSansSC-Regular.otf');

async function ensureFont() {
  if (existsSync(FONT_PATH) && readFileSync(FONT_PATH).length > 1_000_000) {
    return FONT_PATH;
  }
  mkdirSync(FONTS_DIR, { recursive: true });
  console.log('[og] downloading CJK font (one-time, ~16MB)...');
  const res = await fetch(FONT_URL, { redirect: 'follow' });
  if (!res.ok) throw new Error(`Failed to download font: ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  writeFileSync(FONT_PATH, buf);
  console.log(`[og] font cached at ${FONT_PATH}`);
  return FONT_PATH;
}

const COLORS = {
  bg: '#0f0f1a',
  bgSoft: '#1a1a2e',
  accent: '#a855f7',
  accent2: '#6366f1',
  text: '#f8fafc',
  textDim: '#94a3b8',
  textMuted: '#64748b',
  border: 'rgba(168, 85, 247, 0.25)',
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

function buildTemplate({ title, description, category, tags, author, siteName, isArticle }) {
  const tagBadges = (tags || [])
    .slice(0, 3)
    .map(
      (t, i) =>
        `<div style="display:flex;align-items:center;padding:8px 16px;border-radius:999px;background:rgba(168,85,247,${0.1 + i * 0.05});border:1px solid ${COLORS.border};color:#c4b5fd;font-size:22px;font-weight:500;">#${escapeXml(t)}</div>`,
    )
    .join('');

  return `<div style="display:flex;flex-direction:column;width:1200px;height:630px;background:linear-gradient(135deg, ${COLORS.bg} 0%, ${COLORS.bgSoft} 100%);font-family:'Noto';position:relative;"><div style="position:absolute;top:-200px;left:-200px;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle, rgba(102,126,234,0.25) 0%, transparent 70%);display:flex;"></div><div style="position:absolute;bottom:-250px;right:-150px;width:600px;height:600px;border-radius:50%;background:radial-gradient(circle, rgba(168,85,247,0.22) 0%, transparent 70%);display:flex;"></div><div style="position:absolute;inset:0;opacity:0.04;display:flex;flex-direction:column;">${Array.from({ length: 12 })
    .map(
      (_, i) =>
        `<div style="position:absolute;left:0;right:0;top:${i * 55}px;height:1px;background:#fff;display:flex;"></div>`,
    )
    .join('')}${Array.from({ length: 22 })
    .map(
      (_, i) =>
        `<div style="position:absolute;top:0;bottom:0;left:${i * 60}px;width:1px;background:#fff;display:flex;"></div>`,
    )
    .join('')}</div><div style="display:flex;flex-direction:column;flex:1;padding:60px 80px;position:relative;"><div style="display:flex;align-items:center;justify-content:space-between;"><div style="display:flex;align-items:center;gap:16px;"><div style="display:flex;align-items:center;justify-content:center;width:56px;height:56px;border-radius:12px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:#fff;font-size:28px;font-weight:700;">孟</div><div style="display:flex;flex-direction:column;"><div style="color:#f8fafc;font-size:24px;font-weight:600;line-height:1.2;">${escapeXml(siteName)}</div><div style="color:#64748b;font-size:18px;margin-top:2px;">menggq.pages.dev</div></div></div>${
    isArticle && category
      ? `<div style="display:flex;align-items:center;gap:8px;padding:10px 20px;border-radius:8px;background:rgba(168,85,247,0.15);border:1px solid ${COLORS.border};color:#c4b5fd;font-size:20px;font-weight:500;"><div style="display:flex;width:8px;height:8px;border-radius:50%;background:#a855f7;"></div>${escapeXml(category)}</div>`
      : ''
  }</div><div style="display:flex;flex-direction:column;margin-top:auto;margin-bottom:auto;max-width:1040px;"><div style="color:#f8fafc;font-size:${isArticle ? 64 : 72}px;font-weight:700;line-height:1.2;letter-spacing:-0.02em;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;">${escapeXml(truncate(title, isArticle ? 60 : 24))}</div>${
    description && isArticle
      ? `<div style="color:${COLORS.textDim};font-size:26px;line-height:1.5;margin-top:24px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;max-width:1000px;">${escapeXml(truncate(description, 140))}</div>`
      : ''
  }</div><div style="display:flex;align-items:center;justify-content:space-between;margin-top:24px;"><div style="display:flex;gap:12px;flex-wrap:wrap;">${tagBadges}</div><div style="display:flex;align-items:center;gap:10px;color:${COLORS.textMuted};font-size:22px;"><div style="display:flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:#fff;font-size:18px;font-weight:700;">${escapeXml(
    (author || '孟').slice(0, 1),
  )}</div><div style="display:flex;">${escapeXml(author || '孟国庆')}</div></div></div></div></div>`;
}

async function renderOgPng({ fontData, template }) {
  const tree = satoriHtml(template);
  const svg = await satori(tree, {
    width: 1200,
    height: 630,
    fonts: [
      {
        name: 'Noto',
        data: fontData,
        weight: 400,
        style: 'normal',
      },
    ],
  });
  const resvg = new Resvg(svg, {
    fitTo: { mode: 'width', value: 1200 },
    font: {
      loadSystemFonts: false,
    },
  });
  return resvg.render().asPng();
}

function _debugTree(node, depth = 0) {
  if (!node) return;
  const pad = '  '.repeat(depth);
  const tag = node.type || (node.props ? 'el' : typeof node);
  const display = node.props?.style?.display || '';
  const childCount = node.props?.children
    ? Array.isArray(node.props.children)
      ? node.props.children.length
      : 1
    : 0;
  console.log(`${pad}<${tag} display="${display}" children=${childCount}>`);
  const children = node.props?.children;
  if (Array.isArray(children)) {
    children.forEach((c) => _debugTree(c, depth + 1));
  } else if (typeof children === 'object' && children) {
    _debugTree(children, depth + 1);
  }
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
  const fontPath = await ensureFont();
  const fontData = readFileSync(fontPath);

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
      isArticle: job.isArticle !== false,
    });
    const png = await renderOgPng({ fontData, template: tpl });
    const file = safeSlug(job.slug) + '.png';
    const outPath = join(outDir, file);
    writeFileSync(outPath, png);
    count++;
  }
  console.log(`[og] generated ${count} og images -> ${outDir}`);
}

main().catch((err) => {
  console.error('[og] error:', err);
  process.exit(1);
});
