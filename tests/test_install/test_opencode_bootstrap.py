"""Offline archive verification and CPU selection checks for the Docker helper."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix", "Docker bootstrap runs on Linux")
class OpenCodeBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="headroom-opencode-test-")
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.trace = self.root / "trace"
        self.env = dict(
            os.environ,
            PATH=f"{self.bin}:{os.environ['PATH']}",
            HOME=str(self.root / "home"),
            TMPDIR=str(self.root),
            PROBE_TRACE=str(self.trace),
            PROBE_ARCH="x86_64",
            PROBE_AVX2="1",
            PROBE_SYMLINK="0",
        )
        self.stub("uname", 'printf "%s\\n" "$PROBE_ARCH"')
        self.stub("grep", '[ "$PROBE_AVX2" = 1 ]')
        self.stub(
            "curl",
            """printf 'download\\n' >> "$PROBE_TRACE"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output) printf 'untrusted archive fixture' > "$2"; shift 2;;
        https://*) printf '%s\\n' "$1" > "$PROBE_TRACE.url"; shift;;
        *) shift;;
    esac
done""",
        )
        self.stub(
            "tar",
            """printf 'extract\\n' >> "$PROBE_TRACE"
while [ "$#" -gt 0 ]; do
    case "$1" in --directory) destination="$2"; shift 2;; *) shift;; esac
done
if [ "$PROBE_SYMLINK" = 1 ]; then
    ln -s /bin/sh "$destination/opencode"
else
    printf 'inert fixture' > "$destination/opencode"
fi""",
        )
        self.stub(
            "install",
            'printf "install\\n" >> "$PROBE_TRACE"\nprintf "%s\\n" "$@" > "$PROBE_TRACE.args"',
        )
        self.script = Path(__file__).resolve().parents[2] / "scripts/install-opencode.sh"

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def run_bootstrap(self, version="1.17.8"):
        self.assertTrue(self.script.is_file(), "checksum bootstrap helper is required")
        return subprocess.run(
            ["sh", str(self.script), version], env=self.env, capture_output=True, text=True
        )

    def allow_fixture_checksum(self):
        self.stub("sha256sum", 'cat > "$PROBE_TRACE.checksum"')

    def assert_target(self, target):
        self.allow_fixture_checksum()
        result = self.run_bootstrap()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            Path(str(self.trace) + ".url").read_text().strip(),
            f"https://github.com/anomalyco/opencode/releases/download/v1.17.8/opencode-{target}.tar.gz",
        )
        self.assertEqual(self.trace.read_text().splitlines(), ["download", "extract", "install"])
        self.assertEqual(
            Path(str(self.trace) + ".args").read_text().splitlines()[-1],
            str(self.root / "home/.opencode/bin/opencode"),
        )

    def test_corrupt_archive_stops_before_extraction(self):
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sha256sum", result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["download"])
        self.assertFalse((self.root / "home").exists())

    def test_failed_download_stops_before_extraction(self):
        self.stub("curl", "exit 22")
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.trace.exists())
        self.assertEqual(list(self.root.iterdir()), [self.bin])

    def test_x64_without_avx2_keeps_baseline_binary(self):
        self.env["PROBE_AVX2"] = "0"
        self.assert_target("linux-x64-baseline")

    def test_x64_with_avx2_keeps_standard_binary(self):
        self.assert_target("linux-x64")

    def test_arm64_keeps_arm64_binary(self):
        self.env["PROBE_ARCH"] = "aarch64"
        self.assert_target("linux-arm64")

    def test_unsupported_version_stops_before_download(self):
        result = self.run_bootstrap("unreviewed-version")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported", result.stderr)
        self.assertFalse(self.trace.exists())

    def test_unsupported_architecture_stops_before_download(self):
        self.env["PROBE_ARCH"] = "unknown-fixture"
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported", result.stderr)
        self.assertFalse(self.trace.exists())

    def test_symlink_archive_member_is_not_installed(self):
        self.allow_fixture_checksum()
        self.env["PROBE_SYMLINK"] = "1"
        result = self.run_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.trace.read_text().splitlines(), ["download", "extract"])
        self.assertFalse((self.root / "home").exists())


if __name__ == "__main__":
    unittest.main()
