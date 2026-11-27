import { mkdir, readFile, writeFile } from 'node:fs/promises';

const source = new URL('../machine-learning-course.md', import.meta.url);
const docs = new URL('../src/content/docs/', import.meta.url);
const markdown = await readFile(source, 'utf8');
const title = "Machine Learning: Foundations and Practice";
const body = markdown.replace(/^\uFEFF?# [^\r\n]+\r?\n/, '');

await mkdir(docs, { recursive: true });
await writeFile(
  new URL('course.md', docs),
  `---\ntitle: ${JSON.stringify(title)}\n---\n${body}`,
);
