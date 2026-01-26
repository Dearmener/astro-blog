import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { visit } from 'unist-util-visit';

function remarkObsidianImages() {
  return (tree, file) => {
    visit(tree, 'paragraph', (node, index, parent) => {
      if (!node.children || node.children.length !== 1) return;
      
      const child = node.children[0];
      if (child.type !== 'text') return;
      
      const match = child.value.match(/^!\[\[([^\]]+)\]\]$/);
      if (!match) return;
      
      let imagePath = match[1];
      
      if (imagePath.includes('|')) {
        imagePath = imagePath.split('|')[0];
      }
      
      if (imagePath.startsWith('../assets/') || imagePath.startsWith('./assets/')) {
        imagePath = '/assets/' + imagePath.replace(/^\.\.?\/assets\//, '');
      } else if (imagePath.startsWith('assets/')) {
        imagePath = '/assets/' + imagePath.replace(/^assets\//, '');
      }
      
      parent.children[index] = {
        type: 'paragraph',
        children: [{
          type: 'image',
          url: imagePath,
          alt: imagePath.split('/').pop()?.replace(/\.[^.]+$/, '') || '',
        }]
      };
    });
  };
}

function rehypeImagePath() {
  return (tree) => {
    visit(tree, 'element', (node) => {
      if (node.tagName === 'img' && node.properties?.src) {
        const src = node.properties.src;
        const publicMatch = src.match(/(?:\.\.\/)*public\/assets\/(.+)/);
        if (publicMatch) {
          node.properties.src = `/assets/${publicMatch[1]}`;
          return;
        }
        const publicOtherMatch = src.match(/(?:\.\.\/)*public\/(.+)/);
        if (publicOtherMatch) {
          node.properties.src = `/${publicOtherMatch[1]}`;
        }
      }
    });
  };
}

export default defineConfig({
  site: 'https://example.com',
  integrations: [
    tailwind(),
    mdx(),
    sitemap(),
  ],
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
      wrap: true,
    },
    remarkPlugins: [remarkObsidianImages, remarkMath],
    rehypePlugins: [rehypeKatex, rehypeImagePath],
  },
});
