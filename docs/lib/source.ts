import { loader } from 'fumadocs-core/source';
import { docs } from '../.source/server';

export const source = loader({
  baseUrl: '/docs',
  source: docs.toFumadocsSource(),
});

type Page = ReturnType<typeof source.getPages>[number];

export function getPageMarkdownUrl(page: Page) {
  const segments = page.slugs.length > 0 ? [...page.slugs, 'content.md'] : ['index', 'content.md'];

  return {
    segments,
    url: `/llms.mdx/docs/${segments.join('/')}`,
  };
}

export function getPageImage(page: Page) {
  const segments = page.slugs.length > 0 ? [...page.slugs, 'image.png'] : ['index', 'image.png'];

  return {
    segments,
    url: `/og/docs/${segments.join('/')}`,
  };
}

export async function getLLMText(page: Page) {
  try {
    return await page.data.getText('processed');
  } catch {
    return `# ${page.data.title}\n\n${page.data.description ?? ''}`;
  }
}
