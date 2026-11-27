import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://clmntgq.github.io',
  base: '/Machine-Learning-foundation-and-practice',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    mermaid({ theme: 'forest', autoTheme: true, enableLog: false }),
    starlight({
      title: "Machine Learning: Foundations and Practice",
      customCss: ['katex/dist/katex.min.css'],
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/CLMNTGQ/Machine-Learning-foundation-and-practice' },
      ],
      sidebar: [
        { label: 'Overview', slug: '' },
        { label: 'Full course', slug: 'course' },
        { label: 'Presentation', link: 'https://clmntgq.github.io/Machine-Learning-foundation-and-practice/machine-learning.html' },
      ],
    }),
  ],
});
