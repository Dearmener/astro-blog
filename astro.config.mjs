import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { visit } from 'unist-util-visit';

// 自定义 rehype 插件：转换图片路径
// 将 ../../../public/assets/xxx.png 转换为 /assets/xxx.png
function rehypeImagePath() {
  return (tree) => {
    visit(tree, 'element', (node) => {
      if (node.tagName === 'img' && node.properties?.src) {
        const src = node.properties.src;
        // 匹配包含 public/assets/ 的相对路径
        const match = src.match(/(?:\.\.\/)*public\/assets\/(.+)/);
        if (match) {
          node.properties.src = `/assets/${match[1]}`;
        }
      }
    });
  };
}

// https://astro.build/config
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
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex, rehypeImagePath],
  },
});
