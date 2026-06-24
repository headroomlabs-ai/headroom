import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import { Inter } from 'next/font/google';
import type { Metadata } from 'next';

const inter = Inter({
  subsets: ['latin'],
});

// Canonical URL for the live docs. ``metadataBase`` resolves the og:url
// and twitter:url for every page; pointing it at the actual live site
// is what lets crawlers (search + LLM) follow the right canonical and
// pick up ``/llms.txt`` / ``/sitemap.xml`` / og images. Override at
// build time via ``NEXT_PUBLIC_SITE_URL`` (e.g. when promoting to a
// custom domain).
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://headroom-docs.vercel.app';

export const metadata: Metadata = {
  title: {
    default: 'Headroom — Context Optimization Layer for AI Agents',
    template: '%s | Headroom',
  },
  description:
    'Compress everything your AI agent reads — tool outputs, logs, files, RAG chunks. Same answers, fraction of the tokens. Library, proxy, MCP server. Local-first. Apache 2.0.',
  metadataBase: new URL(SITE_URL),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    type: 'website',
    siteName: 'Headroom',
    title: 'Headroom — Context Optimization Layer for AI Agents',
    description:
      'Compress tool outputs, logs, files, and RAG chunks before they reach the LLM. 60–95% fewer tokens, same answers.',
    url: '/',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Headroom — Context Optimization Layer for AI Agents',
    description:
      'Compress tool outputs, logs, files, and RAG chunks before they reach the LLM. 60–95% fewer tokens, same answers.',
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
