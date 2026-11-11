# Immutable candidate version and source isolation

Candidate versions come from `project.version` in the requested commit's
`pyproject.toml`. The trusted workflow producer reads that tree with `git show`
after validating the source and producer ancestry against authoritative main.
It passes the resolved version into the reusable build and uses the same value
for candidate emission and downloaded-byte verification. Candidate detection
does not execute historical version scripts or consult mutable release tags.

Rerunning a SHA retains its version assignment, even if later tags are added or
the producer's checkout has a different version. To build a candidate with a
different version, select a commit that records that version. This guarantees
the version assignment, not bit-for-bit reproducibility across different build
toolchains. The existing release publisher retains its ordinary version logic
when the reusable build's `resolved_version` input is empty.

Supported assignments use canonical Python versions: `X.Y.Z`, `X.Y.ZaN`,
`X.Y.ZbN` or `X.Y.ZrcN`, without leading zeros in numeric components.
Python metadata keeps that assignment; npm artifacts use the corresponding
`X.Y.Z-alpha.N`, `X.Y.Z-beta.N` or `X.Y.Z-rc.N` spelling. Package synchronization
and its verifier run with the Python spelling before the npm packager rewrites
tarball metadata. Candidate provenance continues to bind the Python version.

Authenticated checkout occurs only in trusted preparation jobs. Checkout uses
`persist-credentials: false`; source preparation rejects any retained local
HTTP credential header. Git history is reconstructed from a bundle containing
objects and refs, with the selected commit restored as HEAD. The history archive
uses this clean history instead of the checkout's Git directory, so checkout
configuration, credential-bearing remotes and includes, worktree configuration,
hooks and reflogs are not transported. The original checkout remains untouched.
Source and history are separate artifacts. Only version detection and changelog
generation download history; the five wheel jobs receive source without a Git
directory. Ordinary release detection retains its tags and complete commit graph.
The producer tooling and historical source are transferred as same-run Actions
artifacts to fresh jobs, including private-repository source when authorized.

Version detection, packaging, wheel compilation, wheel smoke imports, candidate
emission, and candidate verification request no repository permissions. They
do not perform an authenticated checkout. Historical rollout code also runs
in that zero-repository-permission boundary. Actions artifact transfer still
uses the runner's artifact service capability; this is not a general sandbox
or a claim that historical code has no network access.

The local replay driver retained with the PR evidence executes the actual
snapshot, restoration and version-detection shell steps against a real Git
repository. It demonstrates retained-header rejection, exact-SHA preservation,
tag-independent version resolution, and bypassing a historical version script
that deliberately raises if executed. It does not exercise hosted token
enforcement, a new five-platform build, production dispatch, or publication.
