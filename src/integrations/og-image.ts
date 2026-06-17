import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import yaml from 'js-yaml';
import type { AstroIntegrationLogger } from 'astro';

interface BuildDonePayload {
  dir: URL;
  logger: AstroIntegrationLogger;
  pages: { pathname: string }[];
  routes: { route: string; type: string }[];
}

/**
 * Astro integration that, after the static build, generates a per-post
 * OG image (1200x630 PNG) under dist/og/<slug>.png so that
 * WeChat / Twitter / Facebook crawlers can pick up a real PNG card
 * (WeChat does not support SVG og:images).
 *
 * The integration shells out to scripts/generate-og.mjs to keep the
 * build hook tiny and the renderer (satori + resvg) out of the
 * integration runtime.
 *
 * Note: we cannot import `astro:content` here because integrations are
 * evaluated in a Node-side context. We instead walk the content
 * directory directly and parse YAML frontmatter.
 */
export default function ogImageIntegration() {
  return {
    name: 'og-image-integration',
    hooks: {
      'astro:build:done': async ({ dir, logger }: BuildDonePayload) => {
        const log = logger.fork('og');

        // Walk src/content/blog to find all posts.
        const contentRoot = join(process.cwd(), 'src', 'content', 'blog');
        const posts = collectPosts(contentRoot);

        const jobs = {
          siteName: '孟国庆的博客',
          posts: [
            // Generic site card for the homepage and non-article pages.
            {
              slug: '__site__',
              title: '孟国庆的博客',
              description: '记录技术 · 分享生活 · 探索 AI',
              category: '',
              tags: ['AI', '机器学习', '深度学习', 'LLM'],
              author: '孟国庆',
              isArticle: false,
            },
            ...posts.map((p) => ({
              slug: p.slug,
              title: p.title,
              description: p.description,
              category: p.category,
              tags: p.tags,
              author: '孟国庆',
              isArticle: true,
            })),
          ],
        };

        // Write the OG PNGs directly into the built dist/ directory so they
        // ship with the static deploy without polluting public/.
        const ogDir = new URL('./og/', dir).pathname;
        const rootOut = dir.pathname;

        log.info(`generating OG images for ${posts.length} posts + site card -> ${ogDir}`);

        const scriptPath = new URL('../../scripts/generate-og.mjs', import.meta.url).pathname;
        const result = spawnSync(
          process.execPath,
          [scriptPath, JSON.stringify(jobs), ogDir],
          {
            stdio: 'inherit',
            env: process.env,
          },
        );

        if (result.status !== 0) {
          throw new Error(`generate-og.mjs exited with status ${result.status}`);
        }

        // Also copy the generic site card to /og-image.png at the site root
        // so non-article pages have a fallback share image.
        const siteCard = `${ogDir}__site__.png`;
        const fallback = `${rootOut}og-image.png`;
        if (existsSync(siteCard)) {
          copyFileSync(siteCard, fallback);
          log.info('copied generic site card to og-image.png');
        }

        log.info('OG images generated successfully');
      },
    },
  };
}

interface CollectedPost {
  slug: string;
  title: string;
  description: string;
  category: string;
  tags: string[];
}

function collectPosts(root: string): CollectedPost[] {
  if (!existsSync(root)) return [];
  const out: CollectedPost[] = [];
  walk(root, (file) => {
    if (!/\.(md|mdx)$/i.test(file)) return;
    const post = parsePost(file, root);
    if (post && post.title) out.push(post);
  });
  return out;
}

function walk(dir: string, onFile: (file: string) => void) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      if (entry === '.obsidian' || entry === 'assets' || entry.startsWith('.')) continue;
      walk(full, onFile);
    } else {
      onFile(full);
    }
  }
}

function parsePost(file: string, root: string): CollectedPost | null {
  let raw: string;
  try {
    raw = readFileSync(file, 'utf-8');
  } catch {
    return null;
  }
  const match = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return null;
  let fm: any;
  try {
    fm = yaml.load(match[1]);
  } catch {
    return null;
  }
  if (!fm || typeof fm !== 'object') return null;
  if (fm.draft === true) return null;
  // Slug: relative path from root, with extension stripped, using OS separator
  const rel = relative(root, file).replace(/\.(md|mdx)$/i, '');
  const slug = rel.split(sep).join('/');
  return {
    slug,
    title: String(fm.title || ''),
    description: String(fm.description || ''),
    category: String(fm.category || '未分类'),
    tags: Array.isArray(fm.tags) ? fm.tags.map(String) : [],
  };
}
