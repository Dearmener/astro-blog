import { readFile } from 'node:fs/promises';
import { readdirSync, statSync } from 'node:fs';
import { join, relative, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import TurndownService from 'turndown';
import { JSDOM } from 'jsdom';

const turndown = new TurndownService({
  headingStyle: 'atx',
  codeBlockStyle: 'fenced',
  bulletListMarker: '-',
});

// Helper: check if a turndown node is an element with given class
function hasClass(node: any, cls: string): boolean {
  return node?.classList?.contains?.(cls) ?? false;
}
function tagName(node: any): string {
  return node?.tagName ?? '';
}

// Replace TOC's h2 heading
turndown.addRule('toc-h2', {
  filter: (node: any) => tagName(node) === 'H2' && node?.textContent?.trim() === '目录',
  replacement: () => '## 目录\n\n',
});

// Convert toc-item spans to numbered list items
turndown.addRule('toc-item', {
  filter: (node: any) => hasClass(node, 'toc-item'),
  replacement: (_: string, node: any) => {
    const title = node?.querySelector?.('.toc-title')?.textContent?.trim() || '';
    const num = node?.querySelector?.('.toc-num')?.textContent?.trim() || '';
    return `${num}. ${title}\n`;
  },
});

// Strip decorative elements
turndown.remove('.chapter-num');
turndown.remove('.cover-eyebrow');
turndown.remove('.cover-sub');

// Convert callout / quote → blockquote
turndown.addRule('callout-quote', {
  filter: (node: any) => hasClass(node, 'callout') || hasClass(node, 'quote'),
  replacement: (content: string) => `> ${content.trim().replace(/\n/g, '\n> ')}\n\n`,
});

// Convert takeaway → labeled blockquote (first line bold)
turndown.addRule('takeaway', {
  filter: (node: any) => hasClass(node, 'takeaway'),
  replacement: (content: string) => {
    const lines = content.trim().split('\n');
    if (lines[0] && !lines[0].startsWith('**')) {
      lines[0] = `**${lines[0]}**`;
    }
    return `> ${lines.join('\n> ')}\n\n`;
  },
});

// Convert exec-summary → blockquote
turndown.addRule('exec-summary', {
  filter: (node: any) => hasClass(node, 'exec-summary'),
  replacement: (content: string) => `> ${content.trim().replace(/\n/g, '\n> ')}\n\n`,
});

// Preserve figure captions as italic text
turndown.addRule('figcaption', {
  filter: (node: any) => tagName(node) === 'FIGCAPTION',
  replacement: (content: string) => `*${content.trim()}*\n\n`,
});

function findHtmlFiles(dir: string, base: string, files: string[] = []): string[] {
  const entries = readdirSync(dir);
  for (const entry of entries) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory() && !entry.startsWith('.') && entry !== 'node_modules') {
      findHtmlFiles(full, base, files);
    } else if (st.isFile() && extname(entry) === '.html') {
      files.push(relative(base, full));
    }
  }
  return files;
}

function extractMeta(html: string): {
  title: string;
  description: string;
  pubDate: Date;
  updatedDate?: Date;
  heroImage?: string;
  category: string;
  tags: string[];
  draft: boolean;
  series?: string;
  seriesOrder?: number;
} {
  const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  const title = titleMatch ? titleMatch[1].trim() : 'Untitled';

  const getMeta = (name: string): string | undefined => {
    const regex = new RegExp(
      `<meta[^>]+(?:name|property)=["']${name}["'][^>]+content=["']([^"']*)["']`,
      'i'
    );
    const match = html.match(regex);
    return match ? match[1] : undefined;
  };

  const description = getMeta('description') || '';
  const pubDateStr = getMeta('pubDate') || getMeta('date') || new Date().toISOString();
  const pubDate = new Date(pubDateStr);
  const updatedDateStr = getMeta('updatedDate');
  const updatedDate = updatedDateStr ? new Date(updatedDateStr) : undefined;
  const heroImage = getMeta('heroImage') || getMeta('og:image');
  const category = getMeta('category') || '未分类';
  const tagsStr = getMeta('tags') || getMeta('keywords') || '';
  const tags = tagsStr.split(',').map((t) => t.trim()).filter(Boolean);
  const draftStr = getMeta('draft');
  const draft = draftStr === 'true' || draftStr === '1';
  const series = getMeta('series');
  const seriesOrderStr = getMeta('seriesOrder');
  const seriesOrder = seriesOrderStr ? parseInt(seriesOrderStr, 10) : undefined;

  return { title, description, pubDate, updatedDate, heroImage, category, tags, draft, series, seriesOrder };
}

function extractBodyHTML(html: string): string {
  // Extract <body> innerHTML and strip the cover section
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*)<\/body>/i);
  let body = bodyMatch ? bodyMatch[1].trim() : html;
  body = body.replace(/<section[^>]*class="[^"]*cover[^"]*"[^>]*>[\s\S]*?<\/section>/gi, '');
  return body;
}

export function htmlBlogLoader() {
  return {
    name: 'html-blog-loader',
    load: async ({ store, logger }: { store: any; logger: any }) => {
      const contentDir = fileURLToPath(new URL('../content/blog', import.meta.url));

      let htmlFiles: string[];
      try {
        htmlFiles = findHtmlFiles(contentDir, contentDir);
      } catch {
        logger.warn('No content/blog directory found for HTML loader');
        return;
      }

      for (const relativePath of htmlFiles) {
        const fullPath = join(contentDir, relativePath);
        const raw = await readFile(fullPath, 'utf-8');
        const meta = extractMeta(raw);

        // Parse with JSDOM, pass body element to turndown (bypasses DOMParser issue)
        const bodyHTML = extractBodyHTML(raw);
        const dom = new JSDOM(`<body>${bodyHTML}</body>`);
        const bodyEl = dom.window.document.body;
        const mdContent = turndown.turndown(bodyEl);

        const slug = relativePath.replace(/\.html$/, '');

        store.set({
          id: slug,
          data: {
            ...meta,
            mdContent,
          },
        });
      }

      logger.info(`Loaded ${htmlFiles.length} HTML blog posts`);
    },
  };
}
