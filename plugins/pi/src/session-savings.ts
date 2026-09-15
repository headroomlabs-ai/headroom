import { readFile, writeFile } from "node:fs/promises";

export interface SessionSavings {
	tokensSaved: number;
	tokensBefore: number;
	tokensAfter: number;
	bytesSaved: number;
	retrievals: number;
}

const EMPTY_SAVINGS: SessionSavings = {
	tokensSaved: 0,
	tokensBefore: 0,
	tokensAfter: 0,
	bytesSaved: 0,
	retrievals: 0,
};

export function sessionSavingsPath(sessionFile: string): string {
	return sessionFile.replace(/(\.[^./]+)?$/, ".headroom-savings.json");
}

function nonnegativeInteger(value: unknown): value is number {
	return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

export async function loadSessionSavings(
	sessionFile: string | undefined,
): Promise<SessionSavings> {
	if (!sessionFile) return { ...EMPTY_SAVINGS };
	try {
		const raw: unknown = JSON.parse(
			await readFile(sessionSavingsPath(sessionFile), "utf8"),
		);
		if (
			typeof raw !== "object" ||
			raw === null ||
			!nonnegativeInteger((raw as SessionSavings).tokensSaved) ||
			!nonnegativeInteger((raw as SessionSavings).tokensBefore) ||
			!nonnegativeInteger((raw as SessionSavings).tokensAfter) ||
			!nonnegativeInteger((raw as SessionSavings).bytesSaved) ||
			!nonnegativeInteger((raw as SessionSavings).retrievals)
		) {
			return { ...EMPTY_SAVINGS };
		}
		const parsed = raw as SessionSavings;
		return {
			tokensSaved: parsed.tokensSaved,
			tokensBefore: parsed.tokensBefore,
			tokensAfter: parsed.tokensAfter,
			bytesSaved: parsed.bytesSaved,
			retrievals: parsed.retrievals,
		};
	} catch {
		return { ...EMPTY_SAVINGS };
	}
}

export async function saveSessionSavings(
	sessionFile: string | undefined,
	savings: SessionSavings,
): Promise<void> {
	if (!sessionFile) return;
	await writeFile(
		sessionSavingsPath(sessionFile),
		`${JSON.stringify(savings)}\n`,
		"utf8",
	);
}
