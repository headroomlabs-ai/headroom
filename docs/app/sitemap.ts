// Next.js App Router sitemap convention (Next 13+). Pulls every page
// out of the Fumadocs ``source`` (same source that backs ``/llms.txt``,
// search, and the OG image generator) and emits a valid sitemap.xml.
//
// Search engines and AI crawlers use this to enumerate every doc page
// without scraping HTML. The ``robots.ts`` route advertises the
// sitemap URL so well-behaved crawlers find it on the first GET.

import type { MetadataRoute } from 'next';
import { source } from '@/lib/source';

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://headroom-docs.vercel.app';

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();

  // Static top-level routes (home page; docs index is covered by the
  // page enumeration below).
  const staticRoutes: MetadataRoute.Sitemap = [
    {
      url: `${SITE_URL}/`,
      lastModified: now,
      changeFrequency: 'weekly',
      priority: 1.0,
    },
  ];

  // Every Fumadocs page (introduction, quickstart, installation,
  // integrations, …). ``page.url`` is the relative URL like
  // ``/docs/quickstart``; ``page.data`` carries the front-matter.
  const docPages: MetadataRoute.Sitemap = source.getPages().map((page) => ({
    url: `${SITE_URL}${page.url}`,
    lastModified: now,
    changeFrequency: 'weekly' as const,
    priority: 0.8,
  }));

  return [...staticRoutes, ...docPages];
}
