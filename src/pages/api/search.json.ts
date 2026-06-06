import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';

export const GET: APIRoute = async () => {
  const blogPosts = await getCollection('blog', ({ data }) => !data.draft);
  const htmlPostsRaw = await getCollection('htmlPosts', ({ data }) => !data.draft);
  const posts = [...blogPosts, ...htmlPostsRaw];

  const searchData = posts.map((post) => ({
    title: post.data.title,
    description: post.data.description,
    slug: post.id,
    category: post.data.category,
    tags: post.data.tags,
  }));

  return new Response(JSON.stringify(searchData), {
    headers: {
      'Content-Type': 'application/json',
    },
  });
};
