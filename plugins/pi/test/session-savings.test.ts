import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
	loadSessionSavings,
	saveSessionSavings,
	sessionSavingsPath,
} from "../src/session-savings.js";

describe("session savings persistence", () => {
	it("stores and restores session totals next to the session file", async () => {
		const directory = await mkdtemp(join(tmpdir(), "headroom-pi-session-"));
		const sessionFile = join(directory, "chat.jsonl");
		await writeFile(sessionFile, "{}\n");

		await saveSessionSavings(sessionFile, {
			tokensSaved: 8593,
			tokensBefore: 40_000,
			tokensAfter: 31_407,
			bytesSaved: 12_000,
			retrievals: 2,
		});

		await expect(loadSessionSavings(sessionFile)).resolves.toEqual({
			tokensSaved: 8593,
			tokensBefore: 40_000,
			tokensAfter: 31_407,
			bytesSaved: 12_000,
			retrievals: 2,
		});
		expect(sessionSavingsPath(sessionFile)).toBe(
			join(directory, "chat.headroom-savings.json"),
		);
		const raw = JSON.parse(
			await readFile(sessionSavingsPath(sessionFile), "utf8"),
		);
		expect(raw.tokensSaved).toBe(8593);
	});

	it("returns empty totals for ephemeral or corrupt session files", async () => {
		const directory = await mkdtemp(join(tmpdir(), "headroom-pi-session-"));
		const sessionFile = join(directory, "broken.jsonl");
		await writeFile(sessionSavingsPath(sessionFile), "{not-json");

		await expect(loadSessionSavings(undefined)).resolves.toEqual({
			tokensSaved: 0,
			tokensBefore: 0,
			tokensAfter: 0,
			bytesSaved: 0,
			retrievals: 0,
		});
		await expect(loadSessionSavings(sessionFile)).resolves.toEqual({
			tokensSaved: 0,
			tokensBefore: 0,
			tokensAfter: 0,
			bytesSaved: 0,
			retrievals: 0,
		});
	});
});
