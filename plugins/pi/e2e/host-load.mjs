import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const temporaryDirectory = await mkdtemp(
	join(tmpdir(), "headroom-pi-host-e2e-"),
);
await mkdir(join(temporaryDirectory, ".omp", "agent"), { recursive: true });
await writeFile(
	join(temporaryDirectory, ".omp", "agent", "config.yml"),
	"setupVersion: 1\n",
);

const ptyProbe = String.raw`
import os
import select
import signal
import json
import sys
import time

host = sys.argv[1]
args = sys.argv[1:]
pid, fd = pty.fork()
if pid == 0:
    os.execvp(host, args)

output = bytearray()
loaded = False
setup_started = False
test_started = False
last_intro_sent = 0
last_escape_sent = 0
last_sent = 0
result_path = os.environ.get("HEADROOM_HOST_E2E_RESULT")
deadline = time.monotonic() + 60
try:
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            loaded = loaded or b"[Extensions]" in output or b"Welcome back!" in output
            setup_started = setup_started or b"Setup step" in output
        now = time.monotonic()
        if not loaded and not setup_started and b"press enter to skip" in output and now - last_intro_sent >= 0.5:
            os.write(fd, b"\r")
            last_intro_sent = now
        if setup_started and not loaded and now - last_escape_sent >= 0.5:
            os.write(fd, b"\x1b")
            last_escape_sent = now
        if loaded and not test_started and now - last_sent >= 0.5:
            os.write(fd, b"/headroom status\r")
            last_sent = now
        if loaded and not test_started and b"health online" in output:
            os.write(fd, b"/headroom-e2e\r")
            test_started = True
        if result_path and os.path.exists(result_path):
            with open(result_path, encoding="utf8") as result_file:
                evidence = json.load(result_file)
            if evidence.get("ok") is True:
                sys.exit(0)
            sys.stderr.write("host integration evidence failed: " + json.dumps(evidence) + "\n")
            sys.exit(1)
    if result_path and os.path.exists(result_path):
        with open(result_path, encoding="utf8") as result_file:
            evidence = json.load(result_file)
        if evidence.get("ok") is True:
            sys.exit(0)
        sys.stderr.write("host integration evidence failed: " + json.dumps(evidence) + "\n")
        sys.exit(1)
    sys.stderr.write("host integration timed out before deterministic evidence completed\n")
    sys.exit(1)
finally:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
`;

function command(command, args, cwd = packageRoot) {
	return execFileSync(command, args, {
		cwd,
		encoding: "utf8",
		stdio: ["ignore", "pipe", "pipe"],
	}).trim();
}

function probeHost(name, executable, args) {
	const version = command(executable, ["--version"]);
	const resultPath = join(
		temporaryDirectory,
		`${name.toLowerCase()}-host-result.json`,
	);
	const result = spawnSync(
		process.env.PYTHON ?? "python3",
		["-u", "-c", "import pty\n" + ptyProbe, executable, ...args],
		{
			cwd: temporaryDirectory,
			env: {
				...process.env,
				HOME: temporaryDirectory,
				XDG_CONFIG_HOME: join(temporaryDirectory, ".config"),
				HEADROOM_HOST_E2E_RESULT: resultPath,
			},
			encoding: "utf8",
			timeout: 65_000,
			maxBuffer: 16 * 1024 * 1024,
		},
	);

	if (result.status !== 0) {
		process.stderr.write((result.stdout ?? "").slice(-5_000));
		process.stderr.write((result.stderr ?? "").slice(-30_000));
		throw new Error(
			`${name} ${version} failed its packed host integration gate`,
		);
	}
	if (!existsSync(resultPath)) {
		throw new Error(
			`${name} ${version} did not write host integration evidence`,
		);
	}
	let evidence;
	try {
		evidence = JSON.parse(readFileSync(resultPath, "utf8"));
	} catch (error) {
		throw new Error(
			`${name} ${version} wrote invalid host integration evidence: ${error}`,
		);
	}
	if (
		evidence.ok !== true ||
		evidence.providerEvidence?.provider !== "headroom-e2e-b" ||
		evidence.providerEvidence?.toolResultCount !== 3 ||
		evidence.providerEvidence?.firstResultCompressed !== true ||
		evidence.providerEvidence?.recentResultsRaw !== true ||
		evidence.exactRawResults !== true ||
		evidence.modelSwitchApplied !== true
	) {
		throw new Error(
			`${name} ${version} returned incomplete host evidence: ${JSON.stringify(evidence)}`,
		);
	}
	process.stdout.write(
		`${name} ${version}: packed extension lifecycle verified ${JSON.stringify(evidence)}\n`,
	);
}

try {
	const extensionSpec = process.env.HEADROOM_EXTENSION_SPEC;
	command("npm", ["init", "--yes"], temporaryDirectory);
	if (extensionSpec) {
		command(
			"npm",
			["install", "--ignore-scripts", "--no-audit", "--no-fund", extensionSpec],
			temporaryDirectory,
		);
	} else {
		const packOutput = command("npm", [
			"pack",
			"--pack-destination",
			temporaryDirectory,
		]);
		const tarballName = packOutput.split(/\r?\n/).at(-1);
		if (!tarballName) throw new Error("npm pack did not report a tarball");
		const tarballPath = join(temporaryDirectory, tarballName);
		command(
			"npm",
			["install", "--ignore-scripts", "--no-audit", "--no-fund", tarballPath],
			temporaryDirectory,
		);
	}

	const extensionPath = join(
		temporaryDirectory,
		"node_modules",
		"headroom-pi",
		"src",
		"index.ts",
	);
	if (!existsSync(extensionPath)) {
		throw new Error(
			`packed extension entry point is missing: ${extensionPath}`,
		);
	}

	const driverPath = join(packageRoot, "e2e", "host-driver.ts");
	if (!existsSync(driverPath)) {
		throw new Error(`host integration driver is missing: ${driverPath}`);
	}

	const extensionArgs = [
		"--no-session",
		"--extension",
		extensionPath,
		"--extension",
		driverPath,
	];
	probeHost("Pi", process.env.PI_BIN ?? "pi", [
		"--offline",
		"--no-extensions",
		...extensionArgs,
	]);
	probeHost("OMP", process.env.OMP_BIN ?? "omp", extensionArgs);
} finally {
	await rm(temporaryDirectory, { recursive: true, force: true });
}
