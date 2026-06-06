import { readFile } from 'node:fs/promises';
import { readdirSync, statSync } from 'node:fs';
import { join, relative, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

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

/**
 * Post-process HTML to remove width constraints from body/css rules.
 * Strips max-width from body/html selectors and removes @page rules
 * (which only apply to print anyway).
 */
function removeWidthConstraints(html: string): string {
  // Remove @page blocks entirely
  html = html.replace(/@page\s*\{[^}]*\}/gi, '');

  // Remove max-width from body selector blocks (handles multi-line)
  html = html.replace(/(body\s*\{[^}]*?)max-width\s*:\s*[^;]+;\s*/gi, '$1');
  html = html.replace(/(html\s*\{[^}]*?)max-width\s*:\s*[^;]+;\s*/gi, '$1');

  // Also handle the @media screen block: remove max-width from body inside it
  html = html.replace(/(@media\s+screen\s*\{[\s\S]*?body\s*\{[^}]*?)max-width\s*:\s*[^;]+;\s*/gi, '$1');

  return html;
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
        const processedHtml = removeWidthConstraints(raw);
        const slug = relativePath.replace(/\.html$/, '');

        store.set({
          id: slug,
          data: {
            ...meta,
            htmlContent: processedHtml,
          },
        });
      }

      logger.info(`Loaded ${htmlFiles.length} HTML blog posts`);
    },
  };
}
