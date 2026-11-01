/**
 * WorkBuddy provider for Oh My Pi (local).
 *
 * Direct pipe — no bridge, no local HTTP server:
 *   omp → this plugin → https://www.codebuddy.ai/v2/chat/completions
 *
 * What it does:
 *   1. Reads the desktop session (OAuth access/refresh token) straight from
 *      the WorkBuddy desktop app's login file — the same file the app uses.
 *      Read per request, so token rotations by the app are always picked up.
 *      The app process itself is not required: the file is the whole contract,
 *      and the plugin refreshes it in place when the token lapses.
 *   2. Fetches /v3/config with the WorkBuddy client identity to get the live
 *      model catalog + modelPromotions, and keeps only the models that are
 *      actually FREE (base price x0.00 or an active factor-0 promotion).
 *      omp refreshes the list through its model cache at startup (24 h TTL);
 *      /workbuddy-refresh re-pulls on demand.
 *   3. Registers the "workbuddy" provider whose streamSimple delegates to
 *      pi-ai's built-in openai-completions implementation, injecting the
 *      desktop session's auth + identity headers per request.
 *   4. Registers each free model's declared thinking surface — exactly the
 *      levels the catalog publishes, never an inferred one — so omp's picker
 *      offers them instead of an empty list.
 *   5. Rewrites the UTC+8 quota-reset timestamp in a 429 message into the
 *      system time zone, keeping the original alongside it (display only —
 *      retry delays come from the `retry-after*` headers, not this text).
 *   6. Appends a short image-capability note to the system prompt of vision
 *      requests. WorkBuddy's own harness emits a "Content filtered" reminder
 *      when *its* picker thinks a model is text-only, and the Hunyuan builds
 *      behind this endpoint were trained on traces containing that line — with
 *      a real image on the wire they would still narrate it from memory. The
 *      note tells them to judge from the content they actually received.
 *
 * Errors, on purpose:
 *   - Login file missing / token absent → explicit error naming the fix.
 *   - Session expired and not refreshable → explicit error naming the fix.
 *
 *   7. Keeps working when the vendor blackholes its own host name. Tencent's
 *      authoritative servers began answering `0.0.0.1` for `www.codebuddy.ai`
 *      on 2026-09-22, which makes every connect hang (the desktop app dies the
 *      same way). The name's own published CNAME still resolves to the EdgeOne
 *      edge that serves it, so that address is followed while TLS and `Host`
 *      stay on the canonical name. The last good free list is also kept on
 *      disk, so a network blip can no longer empty the provider out of omp.
 *
 * No timers, no launcher hooks, no models.yml block.
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs"
import { randomUUID } from "node:crypto"
import { promises as dnsPromises } from "node:dns"
import { isIP } from "node:net"
import { dirname, join } from "node:path"
import { homedir, tmpdir } from "node:os"

import { AssistantMessageEventStream, streamSimple as piStreamSimple } from "@oh-my-pi/pi-ai"
import type { ExtensionAPI, ProviderConfig } from "@oh-my-pi/pi-coding-agent"

import { registerOcfreeProvider } from "./ocfree"
import { awaitLogin, baseForDomain, loginBase, openBrowser, startLogin, type LoginBase } from "./oauth"
import {
	accountId,
	activeAccountId,
	setActiveAccount,
	storedAccounts,
	updateTokens,
	upsertAccount,
	type StoredAccount,
} from "./vault"

// ── Constants (mirror the desktop app's own client identity) ─────────────
const WB_APP_VERSION = "5.2.5"
const CONFIG_PATH = "/v3/config"
const REFRESH_SKEW_MS = 5 * 60 * 1000
const REFRESH_COOLDOWN_MS = 15 * 1000
const CATALOG_TTL_MS = 5 * 60 * 1000
/**
 * How long a saved free-model list may stand in for a live one. Bounded on
 * purpose: free-ness is derived from the catalog's own credits/promotions, so a
 * stale copy is a price claim without fresh proof. Past this age the provider
 * reports the outage instead of a list it can no longer vouch for.
 */
const CATALOG_FALLBACK_MAX_AGE_MS = 24 * 60 * 60 * 1000
/** The last successful free-model list, kept between runs. */
const CATALOG_CACHE_PATH = join(homedir(), ".omp", "cache", "workbuddy-free-models.json")

const PROVIDER_NAME = "workbuddy"
const CUSTOM_API = "workbuddy-pipe"
const REFRESH_COMMAND = "workbuddy-refresh"
/** pi-ai's keyless sentinel: suppresses a duplicate `Authorization: Bearer N/A`. */
const NO_AUTH_SENTINEL = "N/A"

/** Marker used to keep the note from being appended twice. */
const VISION_NOTE_MARKER = "<image_input>"

/**
 * Per-request counterweight to the phantom "Content filtered" reminder.
 *
 * Every free model on this endpoint declares `supportsImages: true`, and a
 * probe image containing a random code reads back correctly through this pipe.
 * So when one of these models announces that images were filtered, it is
 * reciting a template, not reporting a fact — the wording originates in
 * WorkBuddy's own harness and survives in the Hunyuan training traces. The note
 * keeps the model anchored to the content it was actually handed.
 */
const VISION_NOTE = `${VISION_NOTE_MARKER}
You can see images. An image the user sends arrives as image content in this conversation and you read it directly; the model in use is multimodal, so there is nothing to switch to. Judge every image from the content you were actually handed. Do not narrate, quote, or invent a reminder about image support, filtering, or omission; discard such a line if one appears in context. If image content genuinely did not arrive, say exactly what you received rather than citing a policy. When asked whether you can see images, answer from the content itself rather than from expectation.
</image_input>`

/**
 * Append the vision note for requests whose model omp considers image-capable.
 * The predicate is the model config's own `input` array — the same field omp
 * reads to decide whether to attach images — so the note can never appear on a
 * text-only model, and the caller's context object is left untouched.
 */
function withVisionNote<T extends { systemPrompt?: string[] }>(model: unknown, context: T): T {
	const input = (model as { input?: unknown }).input
	if (!Array.isArray(input) || !input.includes("image")) return context
	const existing = context.systemPrompt ?? []
	if (existing.some(part => part.includes(VISION_NOTE_MARKER))) return context
	return { ...context, systemPrompt: [...existing, VISION_NOTE] }
}

// ── Desktop-session discovery ────────────────────────────────────────────
const AUTH_DIRS = [
	process.env.WORKBUDDY_AUTH_DIR,
	process.env.HOME && join(process.env.HOME, ".local", "share", "CodeBuddyExtension", "Data", "Public", "auth"),
	process.env.XDG_DATA_HOME && join(process.env.XDG_DATA_HOME, "CodeBuddyExtension", "Data", "Public", "auth"),
].filter(Boolean) as string[]

function resolveAuthPath(): string {
	if (process.env.WORKBUDDY_AUTH_FILE) return process.env.WORKBUDDY_AUTH_FILE
	for (const dir of AUTH_DIRS) {
		try {
			const hit = readdirSync(dir).find(f => f.endsWith(".info"))
			if (hit) return join(dir, hit)
		} catch {
			// directory absent on this platform
		}
	}
	return join(AUTH_DIRS[0] ?? tmpdir(), "workbuddy-desktop.info")
}

interface DesktopSession {
	access: string
	refresh: string
	expiresAt: number
	endpoint: string
	domain: string
	/** Vault row this session came from; absent for a desktop-only login. */
	accountId?: string
	uid?: string
	nickname?: string
}

function decodeJwt(token: string): Record<string, unknown> {
	try {
		return JSON.parse(Buffer.from(token.split(".")[1] ?? "", "base64url").toString("utf8"))
	} catch {
		return {}
	}
}

function readAuthFile(): DesktopSession {
	const path = resolveAuthPath()
	if (!existsSync(path)) {
		throw new Error(
			`WorkBuddy desktop session not found at ${path}. Sign in to the WorkBuddy desktop app once, then retry.`,
		)
	}
	const raw = JSON.parse(readFileSync(path, "utf8")) as { auth?: Record<string, unknown> }
	const auth = raw.auth ?? {}
	const access = typeof auth.accessToken === "string" ? auth.accessToken : ""
	if (!access) {
		throw new Error("WorkBuddy login file has no accessToken. Sign in to the WorkBuddy desktop app, then retry.")
	}
	const claims = decodeJwt(access)
	const rawDomain = typeof auth.domain === "string" ? auth.domain : typeof claims.iss === "string" ? claims.iss : ""
	const endpoint = process.env.CODEBUDDY_ENDPOINT || baseForDomain(rawDomain)
	const domain = rawDomain
		? rawDomain.replace(/^https?:\/\//, "").replace(/\/+$/, "")
		: endpoint.replace(/^https?:\/\//, "")
	return {
		access,
		refresh: typeof auth.refreshToken === "string" ? auth.refreshToken : "",
		expiresAt: Number(auth.expiresAt ?? (typeof claims.exp === "number" ? claims.exp * 1000 : 0)) || 0,
		endpoint,
		domain,
		...(typeof auth.uid === "string" ? { uid: auth.uid } : typeof claims.sub === "string" ? { uid: claims.sub } : {}),
		...(typeof auth.nickname === "string" ? { nickname: auth.nickname } : {}),
	}
}

/**
 * Every credential a request may use, active first.
 *
 * Two sources feed one list: the provider's own vault (browser sign-in and
 * refreshed tokens) and the desktop app's login file. When both describe the
 * same account the vault row wins — it is the copy this process can refresh —
 * so the desktop file is a discovery input here, not a second account.
 */
function requestAccounts(): DesktopSession[] {
	const sessions: DesktopSession[] = storedAccounts().map(account => ({
		access: account.access,
		refresh: account.refresh,
		expiresAt: account.expiresAt,
		endpoint: account.endpoint,
		domain: account.domain,
		accountId: account.id,
		uid: account.uid,
		...(account.nickname ? { nickname: account.nickname } : {}),
	}))
	try {
		const desktop = readAuthFile()
		const duplicate = sessions.some(
			session => session.uid && desktop.uid && session.uid === desktop.uid && session.domain === desktop.domain,
		)
		if (!duplicate) sessions.push(desktop)
	} catch {
		// no desktop login on this machine: the vault has to carry the request
	}
	if (sessions.length === 0) {
		throw new Error(
			"No WorkBuddy account is available. Run /workbuddy-login to sign in, or start WorkBuddy once and sign in there.",
		)
	}
	// The vault's own order already puts the active account first.
	const active = activeAccountId()
	if (active) sessions.sort((a, b) => (a.accountId === active ? -1 : b.accountId === active ? 1 : 0))
	return sessions
}

/** Auth for a model request: the active account, or the first usable one. */
function readStoredAuth(): DesktopSession {
	return requestAccounts()[0]
}

// ── Token refresh (cross-process safe, same protocol the desktop app uses) ──
let lastRefreshFailedAt = 0

function withAuthLock<T>(fn: () => T): T {
	const lockPath = join(tmpdir(), "workbuddy-auth-refresh.lock")
	const deadline = Date.now() + 5000
	while (existsSync(lockPath)) {
		if (Date.now() > deadline) {
			try {
				writeFileSync(lockPath, "")
			} catch {}
			break
		}
		try {
			if (Date.now() - statSync(lockPath).mtimeMs > 20000) {
				writeFileSync(lockPath, "")
				break
			}
		} catch {}
		Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 50)
	}
	try {
		writeFileSync(lockPath, String(process.pid))
	} catch {}
	try {
		return fn()
	} finally {
		try {
			writeFileSync(lockPath, "")
		} catch {}
	}
}

function persistRefreshed(authPath: string, data: Record<string, unknown>): void {
	withAuthLock(() => {
		const raw = JSON.parse(readFileSync(authPath, "utf8")) as { auth?: Record<string, unknown> }
		const prev = raw.auth ?? {}
		if (prev.accessToken !== data.accessToken && Date.now() < Number(prev.expiresAt ?? 0) - REFRESH_SKEW_MS) {
			return // the desktop app refreshed first; don't clobber its newer token
		}
		const now = Date.now()
		const expiresIn = Number(data.expiresIn ?? 0)
		raw.auth = {
			...prev,
			accessToken: data.accessToken,
			...(data.refreshToken ? { refreshToken: data.refreshToken } : {}),
			...(expiresIn ? { expiresIn } : {}),
			tokenType: (prev.tokenType as string) || "Bearer",
			lastRefreshTime: now,
			...(expiresIn ? { expiresAt: now + expiresIn * 1000 } : {}),
			...(data.refreshExpiresIn
				? { refreshExpiresIn: Number(data.refreshExpiresIn), refreshExpiresAt: now + Number(data.refreshExpiresIn) * 1000 }
				: {}),
		}
		const tmp = `${authPath}.omp-plugin-tmp`
		mkdirSync(dirname(authPath), { recursive: true })
		writeFileSync(tmp, JSON.stringify(raw))
		renameSync(tmp, authPath)
	})
}

async function refreshAuth(session: DesktopSession): Promise<DesktopSession | null> {
	if (!session.refresh) return null
	if (Date.now() - lastRefreshFailedAt < REFRESH_COOLDOWN_MS) return null
	try {
		const res = await endpointFetch(`${session.endpoint}/v2/plugin/auth/token/refresh`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Accept: "application/json",
				Authorization: `Bearer ${session.refresh}`,
			},
			signal: AbortSignal.timeout(8000),
		})
		const body = (await res.json().catch(() => null)) as { code?: number; data?: Record<string, unknown> } | null
		const data = body?.data
		if (!res.ok || !body || body.code !== 0 || typeof data?.accessToken !== "string") {
			lastRefreshFailedAt = Date.now()
			return null
		}
		const access = data.accessToken
		const expiresIn = Number(data.expiresIn ?? 0)
		const expiresAt = expiresIn > 0 ? Date.now() + expiresIn * 1000 : session.expiresAt
		const nextRefresh = typeof data.refreshToken === "string" && data.refreshToken ? data.refreshToken : session.refresh
		const refreshExpiresIn = Number(data.refreshExpiresIn ?? 0)
		if (session.accountId) {
			updateTokens(session.accountId, {
				access,
				refresh: nextRefresh,
				expiresAt,
				...(refreshExpiresIn > 0 ? { refreshExpiresAt: Date.now() + refreshExpiresIn * 1000 } : {}),
			})
			return { ...session, access, refresh: nextRefresh, expiresAt }
		}
		// A desktop-sourced login has no vault row yet: keep the app's file in
		// sync exactly as before, so the two clients never hold different tokens.
		persistRefreshed(resolveAuthPath(), data)
		return readAuthFile()
	} catch {
		lastRefreshFailedAt = Date.now()
		return null
	}
}

// ── Reaching the API when the vendor's own DNS blackholes the name ───────
/**
 * Addresses a resolver answers with when a name is deliberately blackholed.
 * `0.0.0.1` is the value Tencent's authoritative nameservers for codebuddy.ai
 * started publishing for `www.codebuddy.ai` (and the apex) on 2026-09-22. The
 * name resolves fine; TCP then hangs until the request times out.
 */
const BLACKHOLE_ADDRESSES = new Set(["0.0.0.0", "0.0.0.1", "::", "127.0.0.1"])

/** Edge answers are cached: a streaming turn must not re-resolve per delta. */
const EDGE_TTL_MS = 5 * 60 * 1000
const edgeCache = new Map<string, { address: string; at: number }>()

/** Set by the extension entry, so the catalog path needs no UI handle. */
let warn: (message: string) => void = () => {}

/** True when a plain fetch to `host` would hang on a blackholed A record. */
async function isUnreachableName(host: string): Promise<boolean> {
	const addresses = await dnsPromises.resolve4(host).catch(() => [] as string[])
	// No A record at all needs the same treatment: the answer may be AAAA-only,
	// which this machine cannot use on a v4-only path either.
	return addresses.length === 0 || addresses.every(address => BLACKHOLE_ADDRESSES.has(address))
}

/**
 * Follow the name's own published CNAME to an address that still serves it.
 *
 * Blackholing the A record does not remove the zone's CNAME: `www.codebuddy.ai`
 * still answers `resolveCname` with its EdgeOne target, and that target still
 * resolves and terminates TLS for the canonical host name. So the address is
 * read from the vendor's own records — nothing is hard-coded here, and an edge
 * change is picked up by the next lookup once the cache window lapses.
 */
async function resolveEdgeAddress(host: string): Promise<string | null> {
	const cached = edgeCache.get(host)
	if (cached && Date.now() - cached.at < EDGE_TTL_MS) return cached.address
	try {
		for (const cname of await dnsPromises.resolveCname(host)) {
			const addresses = await dnsPromises.resolve4(cname).catch(() => [] as string[])
			const address = addresses.find(candidate => !BLACKHOLE_ADDRESSES.has(candidate))
			if (address) {
				edgeCache.set(host, { address, at: Date.now() })
				return address
			}
		}
	} catch {
		// no CNAME published (or the lookup itself failed): nothing to follow
	}
	return null
}

/**
 * The one `fetch` this provider uses for its endpoint.
 *
 * Normal case: the name resolves to a real address and the request is passed
 * through untouched. Blackholed case: the CNAME's address goes in the URL while
 * `Host` and the TLS server name stay on the canonical host — which is what the
 * desktop session, the certificate and the edge's routing are all built around.
 * A name that stays blackholed *and* has no usable CNAME falls through to the
 * plain fetch, so the failure the caller sees is the real one.
 */
const endpointFetch: typeof fetch = async (input, init) => {
	const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url)
	if (url.protocol !== "https:" || isIP(url.hostname) || !(await isUnreachableName(url.hostname))) {
		return fetch(input as never, init as never)
	}
	const host = url.hostname
	const address = await resolveEdgeAddress(host)
	if (!address) return fetch(input as never, init as never)
	const headers = new Headers(init?.headers ?? undefined)
	headers.set("host", host)
	url.hostname = address
	return fetch(url, { ...init, headers, tls: { serverName: host } } as never)
}

// ── Free-model catalog (authoritative source: /v3/config + modelPromotions) ──
interface WbCatalogModel {
	id: string
	name: string
	maxInputTokens?: number
	maxOutputTokens?: number
	/**
	 * The endpoint's *own* operating length for this model, distinct from the
	 * window it will technically validate against (`maxInputTokens`).
	 *   { defaultLength: 300000, supportedLengths: [300000, 1000000] }
	 * `defaultLength` is what the WorkBuddy app itself starts a session on; the
	 * larger entry in `supportedLengths` is an opt-in the user can raise to.
	 * Absent for models that publish only `maxInputTokens`.
	 */
	contextWindow?: { defaultLength?: number; supportedLengths?: number[] }
	supportsImages?: boolean
	supportsToolCall?: boolean
	disabledMultimodal?: boolean
	credits?: string
	/** Reasoning capability flags, as the catalog spells them. */
	supportsReasoning?: boolean
	/**
	 * Thinking surface. Two shapes appear upstream:
	 *   { supportedEfforts: ["low", "high"], defaultEffort: "high", canDisableThinking } — current
	 *   { effort: "high", summary: "auto" }                                              — legacy, toggle-only
	 * The desktop picker reads exactly these fields (its own VALID_REASONING_EFFORTS
	 * is the same six words omp uses), so they are the authority — not the model id.
	 */
	reasoning?: {
		effort?: string
		defaultEffort?: string
		supportedEfforts?: string[]
		canDisableThinking?: boolean
	}
}

/**
 * Image input capability, straight from the upstream catalog.
 *
 * The gateway marks vision support with `supportsImages`, and revokes it per
 * model with `disabledMultimodal: true`. Both fields are authoritative — the
 * desktop app gates its own image-send path on exactly this pair, so omp's
 * `input` array stays in lock-step with what WorkBuddy itself allows.
 * Absent `supportsImages` is treated as text-only (never claim vision we can't
 * prove: omp would then send images the gateway rejects).
 */
function isVisionModel(m: WbCatalogModel): boolean {
	return m.supportsImages === true && m.disabledMultimodal !== true
}

interface WbPromotion {
	enabled?: boolean
	discount?: { factor?: number | string; discountedCredits?: string }
	schedule?: { validFrom?: string; validUntil?: string }
	modelIds?: string[]
}

let catalogCache: { at: number; models: WbCatalogModel[]; freeIds: string[] } = { at: 0, models: [], freeIds: [] }

function parseCredits(v: unknown): number | null {
	if (typeof v !== "string") return null
	const m = /^x([0-9]+(?:\.[0-9]+)?)$/i.exec(v.trim())
	return m ? Number(m[1]) : null
}

function promoActiveNow(promo: WbPromotion, now = new Date()): boolean {
	if (!promo?.enabled || !promo?.discount) return false
	const factor = Number(promo.discount.factor)
	const discounted = parseCredits(promo.discount.discountedCredits)
	if (factor !== 0 && discounted !== 0) return false
	const s = promo.schedule
	if (!s) return true
	const from = s.validFrom ? new Date(s.validFrom).getTime() : -Infinity
	const until = s.validUntil ? new Date(s.validUntil).getTime() : Infinity
	const t = now.getTime()
	return t >= from && t < until
}

function computeFreeIds(models: WbCatalogModel[], promotions: WbPromotion[] | undefined): string[] {
	const free = new Set<string>()
	for (const m of models) {
		if (parseCredits(m.credits) === 0) free.add(m.id)
	}
	for (const p of promotions ?? []) {
		if (!promoActiveNow(p)) continue
		for (const id of p.modelIds ?? []) {
			const m = models.find(x => x.id === id)
			const c = parseCredits(m?.credits)
			if (m && (c === null || c > 0)) free.add(id) // discounted to 0x
		}
	}
	return [...free]
}

function trace(): string {
	return randomUUID().replace(/-/g, "")
}

function buildHeaders(session: DesktopSession, model: string): Record<string, string> {
	const messageId = trace()
	const traceId = trace()
	const spanId = traceId.slice(0, 16)
	const parentSpanId = traceId.slice(16, 32)
	const h: Record<string, string> = {
		Accept: "application/json, text/plain, */*",
		"Content-Type": "application/json",
		"X-Requested-With": "XMLHttpRequest",
		"X-Agent-Intent": "craft",
		"X-IDE-Type": "WorkBuddy",
		"X-IDE-Name": "WorkBuddy",
		"X-IDE-Version": WB_APP_VERSION,
		"X-Product-Version": WB_APP_VERSION,
		"X-Env-ID": "production",
		"X-Domain": session.domain,
		"X-Product": "WorkBuddy",
		"User-Agent": `WorkBuddy/${WB_APP_VERSION}`,
		"X-Request-ID": messageId,
		"X-Conversation-ID": trace(),
		"X-Conversation-Request-ID": messageId,
		"X-Conversation-Message-ID": messageId,
		"X-Request-Trace-Id": traceId,
		b3: `${traceId}-${spanId}-1-${parentSpanId}`,
		"X-B3-TraceId": traceId,
		"X-B3-ParentSpanId": parentSpanId,
		"X-B3-SpanId": spanId,
		"X-B3-Sampled": "1",
	}
	if (model) h["X-Model-ID"] = model
	return h
}

/** A free-model list plus the moment it was proven free. */
type CatalogSnapshot = { at: number; models: WbCatalogModel[]; freeIds: string[] }

function writeCatalogCache(entry: CatalogSnapshot): void {
	try {
		mkdirSync(dirname(CATALOG_CACHE_PATH), { recursive: true })
		const tmp = `${CATALOG_CACHE_PATH}.tmp`
		writeFileSync(tmp, JSON.stringify(entry))
		renameSync(tmp, CATALOG_CACHE_PATH)
	} catch {
		// a cache write must never fail a request
	}
}

function readCatalogCache(): CatalogSnapshot | null {
	try {
		const raw = JSON.parse(readFileSync(CATALOG_CACHE_PATH, "utf8")) as Partial<CatalogSnapshot>
		if (typeof raw?.at !== "number" || !Array.isArray(raw.models) || !Array.isArray(raw.freeIds)) return null
		return { at: raw.at, models: raw.models, freeIds: raw.freeIds }
	} catch {
		return null
	}
}

async function fetchCatalog(): Promise<{ models: WbCatalogModel[]; freeIds: string[] }> {
	if (Date.now() - catalogCache.at < CATALOG_TTL_MS && catalogCache.models.length) {
		return { models: catalogCache.models, freeIds: catalogCache.freeIds }
	}
	// The active account, from whichever source it came, is the catalog's identity
	// too: promotions are per account and the list must match the one serving.
	const session = readStoredAuth()
	try {
		const res = await endpointFetch(`${session.endpoint}${CONFIG_PATH}`, {
			headers: { ...buildHeaders(session, ""), Accept: "application/json", Authorization: `Bearer ${session.access}` },
			signal: AbortSignal.timeout(8000),
		})
		if (!res.ok) throw new Error(`HTTP ${res.status}`)
		const body = (await res.json()) as { data?: { models?: WbCatalogModel[]; modelPromotions?: WbPromotion[] } }
		const models = body?.data?.models
		if (!Array.isArray(models) || models.length === 0) throw new Error("catalog returned no models")
		const freeIds = computeFreeIds(models, body?.data?.modelPromotions)
		catalogCache = {
			at: Date.now(),
			models: models.filter(m => m.supportsToolCall !== false && !/^(codewise|hunyuan-image|kling|completion)/.test(m.id)),
			freeIds,
		}
		writeCatalogCache(catalogCache)
		return { models: catalogCache.models, freeIds }
	} catch (error) {
		// A transport failure must not erase the provider from omp's model list.
		// The saved list stands in — for a bounded time, since it is a price claim
		// made earlier — and the reason is reported either way.
		const detail = error instanceof Error ? error.message : String(error)
		const saved = readCatalogCache()
		if (saved && Date.now() - saved.at < CATALOG_FALLBACK_MAX_AGE_MS) {
			const ageHours = Math.round((Date.now() - saved.at) / 3_600_000)
			warn(`[workbuddy] catalog fetch failed at ${session.endpoint} (${detail}); using the list saved ${ageHours}h ago`)
			catalogCache = saved
			return { models: saved.models, freeIds: saved.freeIds }
		}
		throw new Error(
			`WorkBuddy catalog fetch failed at ${session.endpoint}${CONFIG_PATH} (${detail}). ` +
				"Nothing this endpoint publishes is reachable from this machine right now.",
		)
	}
}

// ── omp model registration ───────────────────────────────────────────────
/**
 * omp's own effort vocabulary, least → most intensive. Identical to WorkBuddy's
 * own VALID_REASONING_EFFORTS, so no level needs remapping in either direction.
 */
const WB_EFFORT_ORDER = ["minimal", "low", "medium", "high", "xhigh", "max"] as const

interface OmpThinkingConfig {
	mode: "effort"
	efforts: string[]
	defaultLevel?: string
	requiresEffort?: boolean
}

interface OmpModelConfig {
	id: string
	name: string
	api: string
	reasoning: boolean
	thinking?: OmpThinkingConfig
	input: ("text" | "image")[]
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number }
	contextWindow: number
	maxTokens: number
	compat: Record<string, unknown>
}

/**
 * The compat every request through this pipe is sent with.
 *
 * `supportsDeveloperRole: false` — the gateway rejects the OpenAI-spec
 * `developer` role (400 code 11128). `thinkingFormat` / `reasoningContentField`
 * — it reads Zhipu/DeepSeek-style `reasoning_content`. `reasoningDisableMode` —
 * how "off" is expressed when a model really allows it. The two
 * `supportsReasoning*` flags are load-bearing: pi-ai applies thinking params
 * only to a compat that declares them (the built-in catalog sets them per host;
 * a plugin-supplied compat gets no defaults), so without them the selected
 * effort is dropped and no thinking param is sent at all.
 *
 * `alwaysSendMaxTokens` + `maxTokensField` — this gateway requires every call to
 * carry an explicit output cap. With `max_tokens` absent it reserves a large
 * implicit output budget of its own and then rejects **any** prompt past ~664.5k
 * tokens with the same 400 code 11133 ("Invalid request parameters") regardless
 * of what else the request contains — the same body returns 200 once `max_tokens`
 * is present, and past 900k tokens, on the wire (§13 of resources/LINUX-PORT.md).
 * So a long omp session used to die the moment its history crossed that line,
 * with nothing in the request actually wrong. `maxTokensField` has to be named as
 * well: a plugin-supplied compat gets no defaults, and pi-ai computes the cap and
 * then drops it when the wire field is unset. The value is pi-ai's usual OpenAI
 * clamp (min(model cap, 64k)); the endpoint accepts it, and omp's own budgeting
 * already keeps the prompt under its real ceiling.
 */
const WB_COMPAT = {
	supportsDeveloperRole: false,
	thinkingFormat: "zai",
	reasoningDisableMode: "zai-thinking-disabled",
	reasoningContentField: "reasoning_content",
	supportsReasoningParams: true,
	supportsReasoningEffort: true,
	maxTokensField: "max_tokens",
	alwaysSendMaxTokens: true,
}

/**
 * Per-model thinking surface for omp's picker: exactly the levels the catalog
 * declares, in canonical order. Nothing is inferred.
 *
 * The current catalog shape publishes a list (`supportedEfforts`); the legacy one
 * names a single level (`effort`), which is what `deepseek-v4.1-flash` carries,
 * so it gets that one level. The gateway does accept other `reasoning_effort`
 * values for the legacy shape (HTTP 200) but ignores them — three repeats each of
 * low/high/max hold the same 602–962 reasoning-character band at the same
 * 2.2–2.6 s — so offering them would be inventing levels.
 * `defaultEffort` — falling back to the legacy `effort` — is the level the
 * desktop app preselects. `canDisableThinking: false` drives `requiresEffort`;
 * see `noOff` below.
 */
function toThinkingConfig(m: WbCatalogModel): OmpThinkingConfig | undefined {
	if (m.supportsReasoning !== true) return undefined
	const { supportedEfforts, effort } = m.reasoning ?? {}
	const declared = supportedEfforts ?? (effort ? [effort] : [])
	// Canonical order, and the only place effort words outside the vocabulary go.
	const efforts = WB_EFFORT_ORDER.filter(e => declared.includes(e))
	if (efforts.length === 0) return undefined
	const preferred = m.reasoning?.defaultEffort ?? effort
	// Only the catalog's *explicit* "cannot disable thinking" turns omp's `off`
	// chip into a clamp. `onlyReasoning: true` is not used for this: it rides on
	// the legacy `{ effort, summary }` shape, where the endpoint does honour
	// `thinking: {type: "disabled"}` — measured off the wire, see §5 of
	// resources/LINUX-PORT.md. Trusting it clamped `off` into real reasoning.
	const noOff = m.reasoning?.canDisableThinking === false
	return {
		mode: "effort",
		efforts: [...efforts],
		...(preferred && efforts.includes(preferred) ? { defaultLevel: preferred } : {}),
		...(noOff ? { requiresEffort: true } : {}),
	}
}

/**
 * The length omp should budget a session to for this model.
 *
 * Prefer the endpoint's own declared `contextWindow.defaultLength` over the
 * larger window it will merely *accept*. `maxInputTokens` is a validation
 * ceiling, not a serving promise: the gateway hands the upstream one invocation
 * with a ~60 s budget, and when a cold-cache prefill overruns it the call comes
 * back `500 code 10000 "invoke model error"` carrying `retry_policy.retry: true`
 * — which pi-ai then re-sends up to its 6-attempt allowance, so omp's 300 s
 * first-event watchdog fires mid-retry and reports the stall, never the 500.
 *
 * Measured on this endpoint with a real session body (step-5-preview, 12 tools,
 * images inlined, cold cache): 146k prompt tokens → 200 in 17.7 s; 252k → 36.1 s;
 * 372k → 50.1 s; ~437k → **500 at 61.9 s**; ~490k and the full ~520k history →
 * 500 at 63.0 s / 61.9 s. The line is the gateway's clock, not the token count,
 * and it sits just above the 300k the catalog itself declares as this model's
 * default — so the declared default is the operating point that stays inside it.
 * A model that publishes no `contextWindow` keeps its full `maxInputTokens`
 * (deepseek's upstream is fast enough to serve it; see §14 of
 * resources/LINUX-PORT.md).
 */
function toOmpContextWindow(m: WbCatalogModel): number {
	const declared = m.contextWindow?.defaultLength
	if (typeof declared === "number" && declared > 0) return declared
	return m.maxInputTokens ?? 192000
}

function toOmpModels(catalog: WbCatalogModel[], freeIds: string[]): OmpModelConfig[] {
	const free = new Set(freeIds)
	return catalog
		.filter(m => free.has(m.id))
		.map(m => {
			const thinking = toThinkingConfig(m)
			return {
				id: m.id,
				name: m.name || m.id,
				api: CUSTOM_API,
				// The catalog's own capability flag; the app gates its thinking toggle on it too.
				reasoning: m.supportsReasoning === true,
				...(thinking ? { thinking } : {}),
				input: isVisionModel(m) ? (["text", "image"] as const) : (["text"] as const),
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: toOmpContextWindow(m),
				maxTokens: m.maxOutputTokens ?? 64000,
			}
		})
}

async function listFreeModels(): Promise<OmpModelConfig[]> {
	const { models, freeIds } = await fetchCatalog()
	return toOmpModels(models, freeIds)
}

// ── Reset-time display ───────────────────────────────────────────────────
/** `<date> <time> UTC±<offset>` — the only shape this endpoint publishes. */
const RESET_STAMP_PATTERN = /\b(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})\s*(?:UTC|GMT)\s*([+-])(\d{1,2})(?::?(\d{2}))?\b/g

/**
 * Append the system-zone rendering of every absolute reset time in `message`.
 *
 * A 429 from this gateway names its window in its own zone — "your usage will
 * reset at 2026-09-23 06:49:44 UTC+8" — and omp shows that verbatim, hours off
 * from the clock the user is reading. Nothing downstream parses an absolute
 * time out of an error message (delays come from the `retry-after*` headers and
 * the "reset in <n><unit>" phrasings), so this is display only. The original
 * stays in place, the local time goes in brackets, and a system already in that
 * zone gets the message untouched.
 */
function localizeResetTimes(message: string | undefined): string | undefined {
	if (typeof message !== "string") return message // an aborted stream carries no text
	return message.replace(RESET_STAMP_PATTERN, (stamp, date, time, sign, hours, minutes) => {
		const offsetMin = (sign === "-" ? -1 : 1) * (Number(hours) * 60 + Number(minutes ?? 0))
		const at = Date.parse(`${date}T${time}Z`) - offsetMin * 60_000
		if (Number.isNaN(at)) return stamp
		if (-new Date(at).getTimezoneOffset() === offsetMin) return stamp // already local
		const local = new Intl.DateTimeFormat(undefined, {
			year: "numeric",
			month: "2-digit",
			day: "2-digit",
			hour: "2-digit",
			minute: "2-digit",
			second: "2-digit",
			hour12: false,
			timeZoneName: "short",
		}).format(new Date(at))
		return `${stamp} (${local} local)`
	})
}

// ── Account failover ─────────────────────────────────────────────────────
/** Per-account cooldowns. In-process only: a restart re-tries everyone once. */
const accountCooldownUntil = new Map<string, number>()

function accountKey(session: DesktopSession): string {
	return session.accountId ?? `${session.domain}:${session.uid ?? session.access.slice(-16)}`
}

function accountLabel(session: DesktopSession): string {
	return session.nickname ?? session.uid ?? session.domain
}

/** The reset time a gateway message names, in epoch ms, when it names one. */
function resetTimeFrom(message: string | undefined): number | undefined {
	if (!message) return undefined
	const match = new RegExp(RESET_STAMP_PATTERN.source).exec(message)
	if (!match) return undefined
	const [, date, time, sign, hours, minutes] = match
	const offsetMin = (sign === "-" ? -1 : 1) * (Number(hours) * 60 + Number(minutes ?? 0))
	const at = Date.parse(`${date}T${time}Z`) - offsetMin * 60_000
	return Number.isNaN(at) ? undefined : at
}

/**
 * A failure that says something about the *account*, not the request. Those
 * are worth repeating on another account; a 400 (or a stream that already
 * delivered content) is not.
 */
function isFailoverEligible(message: { errorStatus?: number; errorMessage?: string }): boolean {
	const status = message.errorStatus
	if (status === 401 || status === 402 || status === 429 || status === 502 || status === 503 || status === 504) return true
	if (status === 403) {
		return /11140|restricted|unusual activity|request illegal|限制使用|异常行为/i.test(message.errorMessage ?? "")
	}
	return false
}

function failoverCooldownMs(message: { errorStatus?: number; errorMessage?: string }): number {
	const reset = resetTimeFrom(message.errorMessage)
	if (reset && reset > Date.now()) return Math.min(reset - Date.now(), 24 * 60 * 60 * 1000)
	if (message.errorStatus === 429) return 15 * 60_000
	if (message.errorStatus === 402) return 6 * 60 * 60_000
	if (message.errorStatus === 403) return 24 * 60 * 60_000
	return 30 * 60_000
}

/** Events whose arrival means the turn is visibly underway and may not move. */
const CONTENT_EVENT_TYPES = new Set([
	"text_start",
	"text_delta",
	"text_end",
	"thinking_start",
	"thinking_delta",
	"thinking_end",
	"toolcall_start",
	"toolcall_delta",
	"toolcall_end",
	"image_end",
	"done",
])

function assistantError(model: { id: string; api?: string; provider?: string }, text: string) {
	return {
		role: "assistant" as const,
		content: [],
		api: model.api ?? CUSTOM_API,
		provider: model.provider ?? PROVIDER_NAME,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "error" as const,
		errorMessage: localizeResetTimes(text),
		timestamp: Date.now(),
	}
}

// ── streamSimple: the direct pipe ────────────────────────────────────────
type StreamSimpleFn = NonNullable<ProviderConfig["streamSimple"]>

function makeStreamSimple(): StreamSimpleFn {
	return (model, context, options) => {
		const stream = new AssistantMessageEventStream()
		void (async () => {
			let candidates: DesktopSession[]
			try {
				candidates = requestAccounts()
			} catch (error) {
				stream.push({
					type: "error",
					reason: "error",
					error: assistantError(model, error instanceof Error ? error.message : String(error)),
				} as never)
				stream.end()
				return
			}

			let lastError: ReturnType<typeof assistantError> | undefined
			let tried = 0
			for (const candidate of candidates) {
				tried++
				const key = accountKey(candidate)
				// A cooling account is skipped while another option remains. As the last
				// option it still gets a try: a stale cooldown must not turn into "no
				// account could serve this".
				if ((accountCooldownUntil.get(key) ?? 0) > Date.now() && tried < candidates.length) continue

				let session = candidate
				let committed = false
				try {
					if (session.expiresAt && session.expiresAt - Date.now() < REFRESH_SKEW_MS) {
						session = (await refreshAuth(session)) ?? session
					}
					// Refresh is best-effort (it is skipped while cooling down after a
					// failure). An expired token that could not be renewed only produces a
					// bare 401 upstream, so name the real fix here instead.
					if (session.expiresAt && session.expiresAt <= Date.now()) {
						lastError = assistantError(
							model,
							`WorkBuddy account ${accountLabel(session)} has expired and could not be refreshed. Run /workbuddy-login or sign in to WorkBuddy again.`,
						)
						continue
					}

					const upstreamHeaders = buildHeaders(session, model.id)
					upstreamHeaders.Authorization = `Bearer ${session.access}`

					const wbCompat = {
						...((model as { compat?: Record<string, unknown> }).compat ?? {}),
						...((model as { compatConfig?: Record<string, unknown> }).compatConfig ?? {}),
						...WB_COMPAT,
					}
					const pipeModel = {
						...model,
						// `compat` may live as a prototype getter on Model — re-attach it
						// explicitly so the spread doesn't drop it.
						compat: wbCompat,
						compatConfig: wbCompat,
						api: "openai-completions",
						baseUrl: `${session.endpoint}/v2`,
						headers: upstreamHeaders,
					}
					// Delegate to pi-ai's own streamer: SSE parsing, tool calls,
					// reasoning replay, retries — all native. Our auth rides in headers;
					// the sentinel keeps a second Bearer out.
					//
					// `streamSimple` (not the api-specific `streamOpenAICompletions`) is
					// deliberate: it is the entry point that runs pi-ai's option mapping,
					// which is where a `thinking.requiresEffort` model has thinking-off
					// clamped to its lowest effort. Calling the api streamer directly
					// skips that and puts `thinking: {type: "disabled"}` on the wire for
					// models that have no thinking-off mode at all.
					const inner = piStreamSimple(pipeModel as never, withVisionNote(model, context), {
						...(options as Record<string, unknown>),
						apiKey: NO_AUTH_SENTINEL,
						headers: upstreamHeaders,
						// Inert unless the endpoint's own name is blackholed upstream.
						fetch: endpointFetch,
					} as never)

					// Hold the opening events back until the turn is visibly underway. An
					// error here (401/402/429/403-11140) says this account is the problem,
					// and the same request may still succeed on the next one. Once content
					// has been pushed, moving accounts would duplicate a partial answer,
					// so from then on the failure is reported as-is.
					const buffered: unknown[] = []
					let failover = false
					for await (const event of inner) {
						// Error text is the one thing that reaches the user unmodified.
						if (event.type === "error") event.error.errorMessage = localizeResetTimes(event.error.errorMessage)
						if (!committed) {
							if (event.type === "error" && tried < candidates.length && isFailoverEligible(event.error)) {
								lastError = event.error
								accountCooldownUntil.set(key, Date.now() + failoverCooldownMs(event.error))
								warn(
									`[workbuddy] ${accountLabel(session)} failed (${event.error.errorStatus ?? "no status"}); retrying on the next account`,
								)
								failover = true
								break
							}
							buffered.push(event)
							if (CONTENT_EVENT_TYPES.has(event.type)) {
								committed = true
								for (const queued of buffered) stream.push(queued as never)
								buffered.length = 0
							}
							continue
						}
						stream.push(event as never)
					}
					if (failover) continue
					for (const queued of buffered) stream.push(queued as never)
					stream.end()
					return
				} catch (error) {
					const failure = assistantError(model, error instanceof Error ? error.message : String(error))
					if (committed) {
						// The turn already had visible output; replaying it on another
						// account would splice two answers together.
						stream.push({ type: "error", reason: "error", error: failure } as never)
						stream.end()
						return
					}
					lastError = failure
					warn(
						`[workbuddy] ${accountLabel(session)} failed locally (${failure.errorMessage ?? "no detail"}); retrying on the next account`,
					)
					continue
				}
			}

			const final = lastError ?? assistantError(model, "No WorkBuddy account could serve this request.")
			stream.push({ type: "error", reason: "error", error: final } as never)
			stream.end()
		})()
		return stream
	}
}

// ── Extension entry ──────────────────────────────────────────────────────
export default function workbuddyProviderExtension(pi: ExtensionAPI): void {
	warn = message => pi.logger.warn(message)
	// Zen's free ids live in the same plugin: a direct pipe to Zen's
	// OpenAI-compatible endpoint, with a minted session id cached once (see
	// ocfree.ts).
	registerOcfreeProvider(pi)
	pi.registerProvider(PROVIDER_NAME, {
		baseUrl: "https://www.codebuddy.ai/v2",
		api: CUSTOM_API,
		streamSimple: makeStreamSimple(),
		authHeader: false,
		fetchDynamicModels: async () => {
			try {
				return await listFreeModels()
			} catch (error) {
				// Startup must not explode when the session file is missing or the
				// catalog is unreachable — the provider simply has no models until
				// it can be read. Request-time errors carry the precise message.
				pi.logger.warn(`[workbuddy] model list unavailable: ${error instanceof Error ? error.message : error}`)
				return []
			}
		},
	} as ProviderConfig)

	pi.registerCommand(REFRESH_COMMAND, {
		description: "Re-fetch the free WorkBuddy model list (promotions rotate server-side)",
		handler: async (_args, ctx) => {
			try {
				catalogCache = { at: 0, models: [], freeIds: [] }
				const models = await listFreeModels()
			// `input` is what the composer and omp's image guard read, `thinking`
			// is what the picker reads. "(no off)" means upstream has no
			// thinking-off mode — there omp's mandatory `off` chip clamps down.
			const describe = (m: OmpModelConfig): string => {
				const vision = m.input.includes("image") ? "vision" : "text"
				const think = m.thinking
					? `think ${m.thinking.efforts.join("/")}${m.thinking.requiresEffort ? " (no off)" : " (off available)"}`
					: m.reasoning
						? "thinking on/off"
						: "no thinking"
				return `${m.id} [${vision}, ${think}]`
			}
				const names = models.map(describe).join(", ") || "(none free right now)"
				ctx.ui.notify(`WorkBuddy free models (${models.length}): ${names}`, "info")
			} catch (error) {
				ctx.ui.notify(`WorkBuddy refresh failed: ${error instanceof Error ? error.message : String(error)}`, "error")
			}
		},
	})

	pi.registerCommand("workbuddy-login", {
		description: "Sign in to a WorkBuddy account in the browser (no desktop app needed; cn | codebuddy | global | URL)",
		handler: async (args, ctx) => {
			const raw = args.trim()
			const named = raw.toLowerCase()
			const kind: LoginBase = named === "cn" ? "cn" : named === "codebuddy" ? "codebuddy" : "global"
			const base = /^https?:\/\//i.test(raw) ? raw : loginBase(kind)
			try {
				const started = await startLogin(base)
				openBrowser(started.url)
				ctx.ui.notify(
					`WorkBuddy sign-in: open ${started.url} — waiting up to 10 minutes for the browser login.`,
					"info",
				)
				// The poll outlives this command handler on purpose: the user is in a
				// browser, and the TUI must stay usable while they finish there.
				void (async () => {
					try {
						const account = await awaitLogin(started.base, started.state)
						const domain = account.tokens.domain || started.base.replace(/^https?:\/\//, "")
						const endpoint = process.env.CODEBUDDY_ENDPOINT || baseForDomain(domain)
						const stored: StoredAccount = {
							id: accountId(account.uid, domain),
							uid: account.uid,
							...(account.nickname ? { nickname: account.nickname } : {}),
							domain,
							endpoint,
							access: account.tokens.access,
							refresh: account.tokens.refresh,
							expiresAt: account.tokens.expiresAt,
							...(account.tokens.refreshExpiresAt ? { refreshExpiresAt: account.tokens.refreshExpiresAt } : {}),
							source: "oauth",
							addedAt: Date.now(),
						}
						upsertAccount(stored, { makeActive: true })
						catalogCache = { at: 0, models: [], freeIds: [] }
						ctx.ui.notify(`WorkBuddy: signed in as ${account.nickname ?? account.uid}.`, "info")
					} catch (error) {
						ctx.ui.notify(
							`WorkBuddy sign-in failed: ${error instanceof Error ? error.message : String(error)}`,
							"error",
						)
					}
				})()
			} catch (error) {
				ctx.ui.notify(
					`WorkBuddy sign-in could not start: ${error instanceof Error ? error.message : String(error)}`,
					"error",
				)
			}
		},
	})

	pi.registerCommand("workbuddy-accounts", {
		description: "List WorkBuddy accounts and pick the one requests use first",
		handler: async (_args, ctx) => {
			const accounts = storedAccounts()
			if (accounts.length === 0) {
				let desktop: DesktopSession | null = null
				try {
					desktop = readAuthFile()
				} catch {
					// no desktop login either
				}
				ctx.ui.notify(
					desktop
						? `Only the desktop login is available (${accountLabel(desktop)}). Run /workbuddy-login to add another account.`
						: "No WorkBuddy account yet. Run /workbuddy-login to add one.",
					"info",
				)
				return
			}
			const active = activeAccountId()
			const labels = accounts.map(
				account => `${account.id === active ? "●" : "○"} ${account.nickname ?? account.uid} · ${account.domain}`,
			)
			if (!ctx.hasUI) {
				ctx.ui.notify(labels.join("\n"), "info")
				return
			}
			const picked = await ctx.ui.select("WorkBuddy account", labels)
			const index = picked ? labels.indexOf(picked) : -1
			if (index < 0) return
			if (setActiveAccount(accounts[index].id)) {
				catalogCache = { at: 0, models: [], freeIds: [] }
				ctx.ui.notify(`WorkBuddy: now using ${accounts[index].nickname ?? accounts[index].uid}.`, "info")
			}
		},
	})
}
