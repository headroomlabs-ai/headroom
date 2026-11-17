# docs

This is a Next.js application generated with
[Create Fumadocs](https://github.com/fuma-nama/fumadocs).

Run development server:

```bash
npm run dev
# or
pnpm dev
# or
yarn dev
```

Open http://localhost:3000 with your browser to see the result.

## Explore

In the project, you can see:

- `lib/source.ts`: Code for content source adapter, [`loader()`](https://fumadocs.dev/docs/headless/source-api) provides the interface to access your content.
- `lib/layout.shared.tsx`: Shared options for layouts, optional but preferred to keep.

| Route                     | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `app/(home)`              | The route group for your landing page and other pages. |
| `app/docs`                | The documentation layout and pages.                    |
| `app/api/search/route.ts` | The Route Handler for search.                          |

### Fumadocs MDX

A `source.config.ts` config file has been included, you can customise different options like frontmatter schema.

Read the [Introduction](https://fumadocs.dev/docs/mdx) for further details.

## Learn More

To learn more about Next.js and Fumadocs, take a look at the following
resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js
  features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
- [Fumadocs](https://fumadocs.dev) - learn about Fumadocs

## Credentialless SDK qualification

After `npm ci`, run `node scripts/verify-sdk-wire.mjs`. It exercises the locked
OpenAI SDK chat API and `ai.generateText` with the OpenAI Responses provider
against an ephemeral loopback HTTP fixture, using explicit fake credentials.
It does not exercise Headroom adapters or make live provider requests.
To typecheck this proof independently of illustrative MDX `noErrors` settings:

```sh
node node_modules/typescript/bin/tsc --ignoreConfig --allowJs --checkJs --noEmit --skipLibCheck --module nodenext --target es2022 scripts/verify-sdk-wire.mjs
```

Use Node24.22.2 and npm12.2.1 for the qualified installation/build workflow.
The exact `esbuild@0.28.2` install script is approved in `allowScripts`; it checks
or acquires the platform binary. `.npmrc` makes newly introduced script identities
fail until reviewed. The root `fumadocs-mdx` postinstall still runs normally.
