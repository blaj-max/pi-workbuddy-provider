/**
 * Browser sign-in for an account that has no desktop app behind it.
 *
 * The desktop client's own OAuth flow is three public calls:
 *
 *   1. POST {base}/v2/plugin/auth/state?platform=workbuddy  → { state, authUrl }
 *   2. the user opens `authUrl` and signs in
 *   3. GET  {base}/v2/plugin/auth/token?state=…             → access + refresh token
 *
 * Then `GET {base}/v2/plugin/login/account` names the account. Nothing here
 * touches the desktop app, and the state is bound to the base that issued it,
 * so a `codebuddy.ai` login can never be polled against `workbuddy.ai`.
 */
import { spawn } from "node:child_process"
import { setTimeout as delay } from "node:timers/promises"

/** How long one sign-in may stay open. */
export const LOGIN_TIMEOUT_MS = 10 * 60 * 1000
/** How often the pending state is polled. */
const POLL_INTERVAL_MS = 3000

const API_PREFIX = "/v2/plugin"

export type LoginBase = "global" | "codebuddy" | "cn"

/** The sign-in host a short name refers to. */
export function loginBase(kind: LoginBase): string {
	switch (kind) {
		case "cn":
			return "https://www.codebuddy.cn"
		case "codebuddy":
			return "https://www.codebuddy.ai"
		default:
			return "https://www.workbuddy.ai"
	}
}

/** Derive the sign-in/API host from a credential's domain. */
export function baseForDomain(domain: string): string {
	if (domain.includes("workbuddy.ai")) return "https://www.workbuddy.ai"
	if (domain.includes("codebuddy.cn")) return "https://www.codebuddy.cn"
	if (domain.includes("codebuddy.ai")) return "https://www.codebuddy.ai"
	return "https://copilot.tencent.com"
}

export interface OAuthTokens {
	access: string
	refresh: string
	expiresAt: number
	refreshExpiresAt?: number
	domain: string
}

export interface OAuthAccount {
	tokens: OAuthTokens
	uid: string
	nickname?: string
}

export interface StartedLogin {
	base: string
	state: string
	url: string
}

function readJsonObject(value: unknown): Record<string, unknown> | undefined {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined
}

function dataOf(document: Record<string, unknown>): Record<string, unknown> {
	return readJsonObject(document.data) ?? {}
}

function firstString(source: Record<string, unknown>, keys: readonly string[]): string {
	for (const key of keys) {
		const value = source[key]
		if (typeof value === "string" && value.trim() !== "") return value.trim()
		if (typeof value === "number" && Number.isFinite(value)) return String(value)
	}
	return ""
}

/** Absolute timestamp (s or ms) or relative TTL (s) → epoch ms. */
function expiryMs(
	source: Record<string, unknown>,
	absoluteKeys: readonly string[],
	ttlKeys: readonly string[],
): number | undefined {
	for (const key of absoluteKeys) {
		const value = source[key]
		const parsed = typeof value === "number" ? value : Number(value)
		if (Number.isFinite(parsed) && parsed > 0) return parsed > 1e12 ? parsed : parsed * 1000
	}
	for (const key of ttlKeys) {
		const value = source[key]
		const parsed = typeof value === "number" ? value : Number(value)
		if (Number.isFinite(parsed) && parsed > 0) return Date.now() + parsed * 1000
	}
	return undefined
}

/** Ask the upstream for a login state and the URL the user opens. */
export async function startLogin(base: string): Promise<StartedLogin> {
	const normalized = base.replace(/\/+$/, "")
	const response = await fetch(`${normalized}${API_PREFIX}/auth/state?platform=workbuddy`, {
		method: "POST",
		headers: { "Content-Type": "application/json", Accept: "application/json" },
		body: "{}",
		signal: AbortSignal.timeout(30_000),
	})
	const document = readJsonObject(await response.json().catch(() => undefined))
	if (!document) throw new Error(`sign-in service at ${normalized} answered non-JSON (HTTP ${response.status})`)
	const data = dataOf(document)
	const state = firstString(data, ["state", "State"])
	if (!state) throw new Error(`sign-in service at ${normalized} issued no state (HTTP ${response.status})`)
	const url =
		firstString(data, ["authUrl", "auth_url", "url", "Url"]) ||
		`${normalized}/login?state=${encodeURIComponent(state)}`
	return { base: normalized, state, url }
}

/** Open a URL in the user's browser. Failure is not fatal — the URL is also shown. */
export function openBrowser(url: string): void {
	const command =
		process.platform === "darwin" ? ["open", [url]] : process.platform === "win32" ? ["cmd", ["/c", "start", "", url]] : ["xdg-open", [url]]
	try {
		const child = spawn(command[0] as string, command[1] as string[], { detached: true, stdio: "ignore" })
		child.on("error", () => {})
		child.unref()
	} catch {
		// no browser integration: the caller prints the URL
	}
}

/**
 * Poll a login once. Pending is the normal answer until the user finishes in
 * the browser; only an explicit token completes it.
 */
export async function pollLogin(
	base: string,
	state: string,
): Promise<{ done: false } | { done: true; account: OAuthAccount }> {
	const response = await fetch(`${base}${API_PREFIX}/auth/token?state=${encodeURIComponent(state)}`, {
		headers: { Accept: "application/json" },
		signal: AbortSignal.timeout(30_000),
	})
	const document = readJsonObject(await response.json().catch(() => undefined))
	if (!document) return { done: false }
	const code = typeof document.code === "number" ? document.code : -1
	if (code !== 0 && code !== 200) return { done: false }
	const data = dataOf(document)
	const access = firstString(data, ["accessToken", "access_token", "AccessToken"])
	if (!access) return { done: false }
	const tokens: OAuthTokens = {
		access,
		refresh: firstString(data, ["refreshToken", "refresh_token", "RefreshToken"]),
		expiresAt: expiryMs(data, ["expiresAt", "expires_at"], ["expiresIn", "expires_in"]) ?? 0,
		domain: firstString(data, ["domain", "Domain"]),
	}
	const refreshExpiresAt = expiryMs(
		data,
		["refreshExpiresAt", "refresh_expires_at"],
		["refreshExpiresIn", "refresh_expires_in"],
	)
	if (refreshExpiresAt) tokens.refreshExpiresAt = refreshExpiresAt

	let profile: Record<string, unknown> = {}
	try {
		if (!tokens.domain) tokens.domain = base.replace(/^https?:\/\//, "").replace(/^www\./, "")
		const headers: Record<string, string> = { Authorization: `Bearer ${access}`, Accept: "application/json" }
		if (tokens.domain) headers["X-Domain"] = tokens.domain
		const res = await fetch(`${base}${API_PREFIX}/login/account`, { headers, signal: AbortSignal.timeout(15_000) })
		const doc = readJsonObject(await res.json().catch(() => undefined))
		if (doc) profile = dataOf(doc)
	} catch {
		// Profile is display metadata; the tokens are already valid.
	}
	const uid = firstString(profile, ["uid", "Uid", "userId", "UserID"]) || jwtSubject(access)
	const nickname = firstString(profile, ["nickname", "NickName", "ScreenName"])
	return {
		done: true,
		account: {
			tokens,
			uid,
			...(nickname ? { nickname } : {}),
		},
	}
}

/** The JWT's own subject claim, used when the profile endpoint is unavailable. */
function jwtSubject(token: string): string {
	try {
		const payload = JSON.parse(Buffer.from(token.split(".")[1] ?? "", "base64url").toString("utf8")) as Record<
			string,
			unknown
		>
		return typeof payload.sub === "string" ? payload.sub : ""
	} catch {
		return ""
	}
}

/**
 * Poll until the login finishes or the deadline passes. The caller owns the
 * user-facing messages; this only reports the outcome.
 */
export async function awaitLogin(
	base: string,
	state: string,
	deadlineMs = Date.now() + LOGIN_TIMEOUT_MS,
): Promise<OAuthAccount> {
	while (Date.now() < deadlineMs) {
		await delay(POLL_INTERVAL_MS)
		const result = await pollLogin(base, state).catch(() => ({ done: false as const }))
		if (result.done) {
			if (!result.account.uid) throw new Error("sign-in finished but the account id could not be read")
			return result.account
		}
	}
	throw new Error("sign-in timed out; run /workbuddy-login again")
}
