# Repository integration

The sidebar lives under `integrations/claude-code-mod` and is discoverable as
`headroom-sidebar` in the repository marketplace. It is separate from the existing
`headroom` startup-hooks plugin. The companion is a separately installed wheel
with an opt-in `claude_mod` proxy-extension entry point. Enabling the plugin alone
does not enable compression or message capture.

Build the wheel with `python scripts/build-wheel.py`; generated wheels, native
API declarations and test logs are ignored. Source tests and the Linux/Windows
workflow are committed. See VALIDATION.md for exact boundaries between fixture,
real proxy and authenticated native Claude checks.

The extension uses a single bounded private `_logs` adapter. A future public
logger accessor can replace that adapter without changing the API or UI.
