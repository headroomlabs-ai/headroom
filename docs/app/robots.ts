// Next.js App Router robots convention (Next 13+). The default Next
// behaviour allows everything; this file makes the intent explicit so
// AI-bot operators that read an opt-in list (GPTBot, ClaudeBot,
// PerplexityBot, Google-Extended, etc.) see a clear green light, and
// so the sitemap is discoverable.
//
// Headroom docs are open-source documentation we WANT indexed. If a
// future page should be excluded, add it to the ``disallow`` list of
// the relevant rule.

import type { MetadataRoute } from 'next';

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://headroom-docs.vercel.app';

export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      // Bot-specific allows. These names are the literal user-agent
      // strings each operator publishes. Listing them explicitly is
      // the documented way to opt INTO AI-training / AI-search
      // indexing — silence (no rule) is treated as opt-out by some
      // operators (notably Google-Extended).
      { userAgent: 'GPTBot', allow: '/' },
      { userAgent: 'OAI-SearchBot', allow: '/' },
      { userAgent: 'ChatGPT-User', allow: '/' },
      { userAgent: 'ClaudeBot', allow: '/' },
      { userAgent: 'Claude-Web', allow: '/' },
      { userAgent: 'anthropic-ai', allow: '/' },
      { userAgent: 'PerplexityBot', allow: '/' },
      { userAgent: 'Perplexity-User', allow: '/' },
      { userAgent: 'Google-Extended', allow: '/' },
      { userAgent: 'cohere-ai', allow: '/' },
      { userAgent: 'CCBot', allow: '/' },
      { userAgent: 'Applebot-Extended', allow: '/' },
      // Catch-all so traditional search crawlers also see an
      // explicit allow.
      { userAgent: '*', allow: '/' },
    ],
    sitemap: `${SITE_URL}/sitemap.xml`,
    host: SITE_URL,
  };
}
