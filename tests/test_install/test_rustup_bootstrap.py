"""Offline shell bootstrap checks; no vendor installer or network is executed."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix", "Docker bootstrap runs on Linux")
class RustupBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="headroom-rustup-test-")
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.trace = self.root / "trace"
        self.env = dict(
            os.environ,
            PATH=f"{self.bin}:{os.environ['PATH']}",
            TMPDIR=str(self.root),
            PROBE_TRACE=str(self.trace),
            PROBE_ARCH="amd64",
            PROBE_KERNEL="x86_64",
            PROBE_NEON="neon",
            PROBE_ARGS=str(self.root / "args"),
            PROBE_INSTALL_NAME=str(self.root / "installer-name"),
        )
        self.stub("dpkg", 'printf "%s\\n" "$PROBE_ARCH"\n[ -n "$PROBE_ARCH" ]')
        self.stub("uname", 'printf "%s\\n" "$PROBE_KERNEL"')
        self.stub(
            "grep",
            'case "$*" in *"/proc/cpuinfo"*) printf "Features : %s\\n" "$PROBE_NEON";; '
            '*) exec /bin/grep "$@";; esac',
        )
        self.stub(
            "curl",
            """printf 'download\\n' >> "$PROBE_TRACE"
out=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output) out="$2"; shift 2;;
        https://*) printf '%s\\n' "$1" > "$PROBE_TRACE.url"; shift;;
        *) shift;;
    esac
done
cat > "$out" <<'INSTALLER'
#!/bin/sh
printf '%s' "${0##*/}" > "$PROBE_INSTALL_NAME"
printf 'executed\\n' >> "$PROBE_TRACE"
printf '%s\\n' "$@" > "$PROBE_ARGS"
INSTALLER
""",
        )
        self.script = Path(__file__).resolve().parents[2] / "scripts/install-rustup.sh"

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def run_bootstrap(self, *args):
        self.assertTrue(self.script.is_file(), "checksum bootstrap helper is required")
        return subprocess.run(
            ["sh", str(self.script), *args], env=self.env, text=True, capture_output=True
        )

    def allow_fixture_checksum(self):
        # Platform/argument tests execute our inert fixture, never vendor bytes.
        self.stub("sha256sum", 'cat > "$PROBE_TRACE.checksum"')

    def assert_target(self, expected):
        self.allow_fixture_checksum()
        result = self.run_bootstrap("-y", "--default-toolchain", "1.95.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            Path(str(self.trace) + ".url").read_text().strip(),
            f"https://static.rust-lang.org/rustup/archive/1.29.3/{expected}/rustup-init",
        )
        self.assertEqual(self.trace.read_text().splitlines(), ["download", "executed"])

    def test_bad_archive_is_rejected_before_execution(self):
        result = self.run_bootstrap("-y")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sha256sum", result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["download"])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["bin", "trace", "trace.url"])

    def test_failed_download_never_executes_installer(self):
        self.stub("curl", "exit 22")
        result = self.run_bootstrap("-y")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.trace.exists())
        self.assertEqual(list(self.root.iterdir()), [self.bin])

    def test_userland_architecture_wins_over_64_bit_kernel(self):
        self.env["PROBE_ARCH"] = "i386"
        self.assert_target("i686-unknown-linux-gnu")

    def test_arm_without_neon_uses_supported_fallback(self):
        self.env.update(PROBE_ARCH="armhf", PROBE_NEON="")
        self.assert_target("arm-unknown-linux-gnueabihf")

    def test_manylinux_without_dpkg_uses_kernel_architecture(self):
        self.env["PROBE_ARCH"] = ""
        self.assert_target("x86_64-unknown-linux-gnu")

    def test_unknown_architecture_stops_before_download(self):
        self.env["PROBE_ARCH"] = "unknown-fixture"
        result = self.run_bootstrap("-y")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported", result.stderr)
        self.assertFalse(self.trace.exists())

    def test_toolchain_options_are_forwarded_verbatim(self):
        self.allow_fixture_checksum()
        args = [
            "-y",
            "--no-modify-path",
            "--profile",
            "minimal",
            "-c",
            "rustfmt",
            "-c",
            "clippy",
            "--default-toolchain",
            "1.95.0",
        ]
        result = self.run_bootstrap(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(self.env["PROBE_ARGS"]).read_text().splitlines(), args)

    def test_binary_keeps_rustup_init_dispatch_name(self):
        # rustup-init selects installer/proxy mode from argv[0], before options.
        self.allow_fixture_checksum()
        result = self.run_bootstrap("-y")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(self.env["PROBE_INSTALL_NAME"]).read_text(), "rustup-init")


if __name__ == "__main__":
    unittest.main()
