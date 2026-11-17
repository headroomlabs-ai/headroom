# Commitlint action provenance

`action.yml` is vendored from [wagoid/commitlint-github-action](https://github.com/wagoid/commitlint-github-action/blob/b948419dd99f3fd78a6548d48f94e3df7f6bf3ed/action.yml) at `b948419dd99f3fd78a6548d48f94e3df7f6bf3ed` (v6.2.1). The upstream MIT license is preserved in `LICENSE`.

The only metadata change replaces the mutable `docker://wagoid/commitlint-github-action:6.2.1` reference with the immutable image digest observed in completed CI job `128302466922`, run `42339062117`:

```text
sha256:86a04e0a99128551a7555c269d2b675c3c85f61358cf7dd558f6b873b66f561a
```

All upstream inputs, defaults, outputs, and Docker execution metadata are preserved. Keeping action metadata preserves GitHub input default expansion and `INPUT_*` environment variables; using an image directly in the workflow would omit that contract. The workflow checkout must precede this local action.

This pins historical execution provenance; it does not attest to image contents or authorize CI execution. Future upgrades must review both the upstream metadata and immutable container digest.
