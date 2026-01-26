import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ 
    pattern: '**/*.{md,mdx}',
    base: './src/content/blog',
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    heroImage: z.string().optional(),
    category: z.string().default('未分类'),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
    series: z.string().optional(),
    seriesOrder: z.number().optional(),
  }),
});

export const collections = { blog };
