---
name: headroom-link-shim
description: "No-op hook shim so local plugin source paths are also valid OpenClaw hook-pack paths."
metadata:
  {
    "openclaw":
      {
        "emoji": "🪝",
        "events": ["command"],
      },
  }
---

# Headroom Link Shim

This hook intentionally does nothing.

OpenClaw currently falls back to validating local plugin paths as hook packs when a
plugin install cannot proceed, such as when the plugin is already installed.
Including this no-op hook keeps `--link` installs from reporting a misleading
`package.json missing openclaw.hooks` error for valid Headroom plugin paths.
