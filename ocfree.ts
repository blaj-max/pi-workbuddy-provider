/**
 * OpenCode Zen free models for omp — direct to the inference endpoint.
 *
 * Zen's free ids are gated to the OpenCode client. The gate is a handshake, not
 * a proxy: the gateway wants a streaming chat request that carries OpenCode's
 * client headers and a `ses_...` session id, and at least one tool. Everything
 * else — which tools, which messages, which system prompt — is ours.
 *
 * So this provider talks to `https://opencode.ai/inference/openai/v1` itself and
 * delegates the wire format to pi-ai's own OpenAI-completions streamer. Native
 * function calling, image inputs, usage, reasoning and aborts are therefore the
 * ordinary omp behaviour, not a text-in/text-out shim:
 *
 *   omp → this provider (Zen headers) → opencode.ai/inference/openai/v1
 *
 * The one thing OpenCode is still needed for is the session id: only its own
 * generator (an HMAC-ish id minted locally) satisfies the gate. A headless
 * `opencode serve` is booted for exactly one `POST /session` (~2 s, once), then
 * killed on the spot. The id is cached in ~/.omp/cache/ocfree/session.json and
 * reused by later runs — it keeps working after the server that minted it is
 * gone, so there is no idle timer, no adoption and no long-lived process.
 *
 * Every request carries OpenCode's client fingerprint (`x-opencode-client`,
 * `x-opencode-session`, `user-agent`) because that is what the free tier
 * checks. This is the interop the previous serve-based pipe got implicitly by
 * routing the whole conversation through OpenCode; the cost here is one tiny
 * local request instead of an OpenCode agent in the middle.
 *
 * Known shape of the gate, measured against the live endpoint:
 *   - `stream: true` and a non-empty `tools` array are required;
 *   - a tool named `read` must be present (omp's tool set has one);
 *   - `x-opencode-session` *and* `x-opencode-session-id` must be a real
 *     `ses_...` id from opencode; a random one is refused with 403;
 *   - the system prompt and conversation are free-form.
 *
 *   OPENCODE_API_KEY     Zen key; omitted → the key in omp's own store is used
 *   OCFREE_OPENCODE_BIN  path to the opencode binary (default: PATH lookup)
 *   OCFREE_SESSION_ID    use this session id and never mint one
 *   OCFREE_STATE_DIR     state/log directory (default ~/.omp/cache/ocfree)
 */
import { spawn } from "node:child_process"
import type { ChildProcess } from "node:child_process"
import { appendFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs"
import { homedir, tmpdir } from "node:os"
import { join } from "node:path"
import { setTimeout as sleep } from "node:timers/promises"

import { streamSimple as piStreamSimple } from "@oh-my-pi/pi-ai"
// The class is exported as a value at runtime but only re-exported as a type
// from the package root, so the concrete subpath is what typechecks.
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream"
import type { AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions, ToolCall } from "@oh-my-pi/pi-ai"
import type { ExtensionAPI, ProviderConfig, ProviderModelConfig } from "@oh-my-pi/pi-coding-agent"

const PROVIDER = "ocfree"
/**
 * Marker API name; streamSimple below implements it, so it is never dispatched.
 * Kept at its old value on purpose: omp caches the registered model list (api
 * name included) in models.db, and renaming it makes a cached entry point at an
 * api nothing serves ("Unhandled API in mapOptionsForApi").
 */
const API = "ocfree-serve"
/** Endpoint override for tests or an alternate host; production is Zen itself. */
const ZEN_BASE = process.env.OCFREE_ZEN_BASE?.replace(/\/+$/, "") ?? "https://opencode.ai/inference/openai/v1"
/**
 * OpenCode's client fingerprint. The `ai-sdk/provider-utils` and `runtime/bun`
 * parts are what the installed client sends; the free tier rejects a request
 * whose user agent is not OpenCode's.
 */
const CLIENT_UA = "opencode/1.19.1 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
const BOOT_TIMEOUT_MS = 25_000
const STATE_DIR = process.env.OCFREE_STATE_DIR ?? join(homedir(), ".omp", "cache", "ocfree")
const SESSION_PATH = join(STATE_DIR, "session.json")
const LOG_PATH = join(STATE_DIR, "server.log")
/** OpenCode's own cwd for the throwaway session — never the user's project. */
const SCRATCH = join(tmpdir(), "ocfree-workspace")
/** OCFREE_DEBUG=1 appends every request decision to the state dir. */
const DEBUG = process.env.OCFREE_DEBUG === "1"

function debug(message: string): void {
	if (!DEBUG) return
	try {
		mkdirSync(STATE_DIR, { recursive: true })
		appendFileSync(join(STATE_DIR, "debug.log"), `${new Date().toISOString()} ${message}\n`)
	} catch {}
}

/**
 * The free ids Zen publishes through the opencode provider, with the catalog's
 * own limits (read from `GET /config/providers` on the running server). Zen
 * rotates this list, so it is worth re-reading when a request starts failing
 * with an unknown model.
 */
const FREE_IDS: Array<{
	id: string
	name: string
	contextWindow: number
	maxTokens: number
	images: boolean
}> = [
	{ id: "deepseek-v4.1-flash-free", name: "DeepSeek V4.1 Flash free (Zen)", contextWindow: 200_000, maxTokens: 32_000, images: true },
	{ id: "big-pickle", name: "Big Pickle free (Zen)", contextWindow: 200_000, maxTokens: 32_000, images: false },
	{ id: "beam-free", name: "Beam free (Zen)", contextWindow: 1_048_576, maxTokens: 131_072, images: false },
	{ id: "glm-5.3-flash-free", name: "GLM-5.3-Flash free (Zen)", contextWindow: 200_000, maxTokens: 32_000, images: true },
	{ id: "longcat-2.5-preview-free", name: "LongCat 2.5 Preview free (Zen)", contextWindow: 1_000_000, maxTokens: 131_072, images: true },
	{ id: "mimo-v2.6-flash-free", name: "MiMo V2.6 Flash free (Zen)", contextWindow: 200_000, maxTokens: 32_000, images: true },
	{ id: "minimax-m3.1-flash-free", name: "MiniMax M3.1 Flash free (Zen)", contextWindow: 1_000_000, maxTokens: 512_000, images: true },
	{ id: "nemotron-3-ultra-free", name: "Nemotron 3 Ultra free (Zen)", contextWindow: 1_000_000, maxTokens: 128_000, images: false },
	{ id: "nemotron-3.5-lightning-free", name: "Nemotron 3.5 Lightning free (Zen)", contextWindow: 262_144, maxTokens: 262_144, images: false },
	{ id: "nemotron-3.5-super-vl-free", name: "Nemotron 3.5 Super VL free (Zen)", contextWindow: 1_048_576, maxTokens: 131_072, images: true },
]

const FREE_MODELS: ProviderModelConfig[] = FREE_IDS.map(entry => ({
	id: entry.id,
	name: entry.name,
	// Reasoning text arrives as `reasoning_content` and is surfaced as thinking;
	// there is no separate effort control to declare.
	reasoning: true,
	input: (entry.images ? ["text", "image"] : ["text"]) as ("text" | "image")[],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: entry.contextWindow,
	// The catalog's output limit is billing metadata, not a sane per-turn cap:
	// omp budgets against this number, so keep it a normal completion size.
	maxTokens: Math.min(entry.maxTokens, 32_768),
}))

// ── Binary + key resolution ──────────────────────────────────────────────
function resolveBinary(): string {
	const fromEnv = process.env.OCFREE_OPENCODE_BIN
	if (fromEnv) return fromEnv
	for (const dir of (process.env.PATH ?? "").split(":")) {
		if (!dir) continue
		const candidate = join(dir, "opencode")
		try {
			if (existsSync(candidate) && statSync(candidate).isFile()) return candidate
		} catch {
			// unreadable PATH entry
		}
	}
	for (const candidate of [join(homedir(), ".npm-global", "bin", "opencode"), join(homedir(), ".bun", "bin", "opencode")]) {
		if (existsSync(candidate)) return candidate
	}
	throw new Error(
		"opencode is not installed (or OCFREE_OPENCODE_BIN is wrong). Install it with `npm i -g opencode-ai`, then retry.",
	)
}

/** Read-only access to omp's own credential store, without logging the value. */
async function storedZenKey(): Promise<string | undefined> {
	try {
		// Indirection keeps TypeScript from demanding bun's ambient types.
		const moduleName = "bun:sqlite"
		const { Database } = (await import(moduleName)) as unknown as {
			Database: new (path: string, options?: { readonly?: boolean }) => {
				query(sql: string): { get(): { data?: string } | null }
				close(): void
			}
		}
		const db = new Database(join(homedir(), ".omp", "agent", "agent.db"), { readonly: true })
		try {
			const row = db
				.query(
					"select data from auth_credentials where provider='opencode-zen' and credential_type='api_key' order by updated_at desc limit 1",
				)
				.get()
			const key = row?.data ? (JSON.parse(row.data) as { key?: unknown }).key : undefined
			return typeof key === "string" && key ? key : undefined
		} finally {
			db.close()
		}
	} catch {
		// Not running under bun, or no stored credential.
		return undefined
	}
}

/**
 * The Zen key, in order: the environment, then omp's own credential store — the
 * same `opencode-zen` credential omp uses. Never logged.
 */
async function resolveZenKey(): Promise<string | undefined> {
	return process.env.OPENCODE_API_KEY?.trim() || (await storedZenKey())
}

// ── Session id (the free-tier handshake) ─────────────────────────────────
interface CachedSession {
	id?: string
	mintedAt?: number
}

let cachedSession: CachedSession | undefined

function readCachedSession(): CachedSession {
	if (cachedSession) return cachedSession
	try {
		const parsed = JSON.parse(readFileSync(SESSION_PATH, "utf8")) as CachedSession
		cachedSession = parsed
	} catch {
		cachedSession = {}
	}
	return cachedSession
}

function writeCachedSession(id: string): void {
	cachedSession = { id, mintedAt: Date.now() }
	try {
		mkdirSync(STATE_DIR, { recursive: true })
		writeFileSync(SESSION_PATH, `${JSON.stringify(cachedSession, null, "\t")}\n`)
	} catch {
		// the cache is an optimisation; a missing one just costs a boot
	}
}

function isValidSessionId(id: string | undefined): id is string {
	return typeof id === "string" && /^ses_[0-9A-Za-z]{20,40}$/.test(id)
}

/** Drop the cached id so the next request mints a fresh one. */
export function forgetOcfreeSession(): void {
	cachedSession = {}
	try {
		rmSync(SESSION_PATH, { force: true })
	} catch {}
}

// ── opncode serve: boot, mint, kill ──────────────────────────────────────
let minting: Promise<string> | undefined
/** Pids this process started — tracked from spawn time so a cut-short mint is cleaned up. */
const owned = new Set<number>()
let exitHookInstalled = false

function pidAlive(pid: number): boolean {
	if (!Number.isInteger(pid) || pid <= 0) return false
	try {
		process.kill(pid, 0)
		return true
	} catch {
		return false
	}
}

function killQuietly(pid: number, signal: NodeJS.Signals = "SIGTERM"): void {
	try {
		process.kill(pid, signal)
	} catch {}
}

function installExitHook(): void {
	if (exitHookInstalled) return
	exitHookInstalled = true
	process.once("exit", () => {
		for (const pid of owned) killQuietly(pid)
		owned.clear()
	})
}

function tail(text: string): string {
	return text
		.replace(/\u001b\[[0-9;]*m/g, "")
		.trim()
		.split("\n")
		.slice(-6)
		.join("\n")
}

async function healthy(url: string): Promise<boolean> {
	try {
		const response = await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(4000) })
		return response.ok
	} catch {
		return false
	}
}

/** Boot `opencode serve`, wait for its banner, and hand back the URL + child. */
async function bootServer(): Promise<{ url: string; child: ChildProcess }> {
	const bin = resolveBinary()
	mkdirSync(STATE_DIR, { recursive: true })
	mkdirSync(SCRATCH, { recursive: true })
	const key = await resolveZenKey()
	const env = { ...process.env, NO_COLOR: "1", ...(key ? { OPENCODE_API_KEY: key } : {}) }
	try {
		writeFileSync(LOG_PATH, `${new Date().toISOString()} spawn ${bin} serve (session mint)\n`)
	} catch {}

	const child = spawn(bin, ["serve", "--pure", "--hostname", "127.0.0.1", "--log-level", "ERROR"], {
		cwd: SCRATCH,
		env,
		stdio: ["ignore", "pipe", "pipe"],
	})
	if (child.pid) owned.add(child.pid)
	installExitHook()

	let output = ""
	const url = await new Promise<string>((resolve, reject) => {
		const timer = setTimeout(() => {
			reject(new Error(`opencode serve did not report a port within ${BOOT_TIMEOUT_MS} ms.\n${tail(output)}`))
		}, BOOT_TIMEOUT_MS)
		const onData = (chunk: Buffer) => {
			output += chunk.toString()
			if (output.length > 64_000) output = output.slice(-32_000)
			const match = output.match(/listening on (https?:\/\/[^\s:]+:\d+)/)
			if (match?.[1]) {
				clearTimeout(timer)
				resolve(match[1])
			}
		}
		child.stdout?.on("data", onData)
		child.stderr?.on("data", onData)
		child.once("error", error => {
			clearTimeout(timer)
			reject(new Error(`cannot run ${bin}: ${error.message}`))
		})
		child.once("exit", code => {
			clearTimeout(timer)
			reject(new Error(`opencode serve exited with code ${code}.\n${tail(output)}`))
		})
	})

	if (!(await healthy(url))) {
		killQuietly(child.pid ?? 0)
		throw new Error(`opencode serve reported ${url} but /api/health did not answer.\n${tail(output)}`)
	}
	return { url, child }
}

/** Ask a throwaway `opencode serve` for one session id, then kill it. */
async function mintSessionId(): Promise<string> {
	const { url, child } = await bootServer()
	const pid = child.pid ?? 0
	try {
		const response = await fetch(`${url}/session?directory=${encodeURIComponent(SCRATCH)}`, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: "{}",
			signal: AbortSignal.timeout(15_000),
		})
		if (!response.ok) throw new Error(`opencode session mint failed: HTTP ${response.status}`)
		const session = (await response.json()) as { id?: string }
		if (!isValidSessionId(session.id)) throw new Error("opencode did not return a usable session id")
		writeCachedSession(session.id)
		return session.id
	} finally {
		// The id outlives the process that minted it — this is why nothing has to
		// stay running between requests.
		owned.delete(pid)
		killQuietly(pid)
		const reap = setTimeout(() => {
			if (pidAlive(pid)) killQuietly(pid, "SIGKILL")
		}, 3000)
		reap.unref?.()
		child.once("exit", () => clearTimeout(reap))
	}
}

/**
 * The session id for a request: `OCFREE_SESSION_ID`, else the cached one, else a
 * freshly minted one. Minting is shared, so concurrent first requests boot one
 * server, not two.
 */
async function ensureSessionId(): Promise<string> {
	const forced = process.env.OCFREE_SESSION_ID?.trim()
	if (forced) return forced
	const cached = readCachedSession().id
	if (isValidSessionId(cached)) {
		debug(`session: using cached ${cached}`)
		return cached
	}
	if (minting) return minting
	debug("session: minting")
	const attempt = mintSessionId()
	minting = attempt
	try {
		return await attempt
	} finally {
		if (minting === attempt) minting = undefined
	}
}

// ── streamSimple: native OpenAI-completions through omp's own streamer ───
function zenHeaders(sessionId: string): Record<string, string> {
	return {
		"user-agent": CLIENT_UA,
		"x-opencode-client": "cli",
		// Both spellings are checked by the gateway; both must be the real id.
		"x-opencode-session": sessionId,
		"x-opencode-session-id": sessionId,
		"x-opencode-project": "global",
	}
}

/** The free tier's client gate: the gateway answers 403 for anything it does not like. */
function isHandshakeError(message: string): boolean {
	return /FreeTierError|free tier can only be used from within OpenCode/i.test(message)
}

/**
 * A connection the gateway killed underneath the stream. Bun words it "The
 * socket connection was closed unexpectedly" (and suggests `fetch(verbose)`),
 * which reads like a bug in this lane but is a transport reset on Zen's side:
 * it hit three turns in a row in one live session (2026-10-22) while 26 probe
 * requests — sequential, parallel, 308 KB bodies — finished cleanly, so it is
 * intermittent, not a property of the request shape. The same message appears
 * when a pooled keep-alive socket turns out to be dead, which a fresh attempt
 * fixes outright.
 */
function isTransportClose(message: string): boolean {
	return /socket connection was closed unexpectedly|socket hang up|other side closed|ECONNRESET|EPIPE|connection (?:was )?(?:reset|closed)|network error|fetch failed|closed before a finish_reason/i.test(
		message,
	)
}

/**
 * Say what happened in the carrier, not the raw runtime text. Reached only after
 * the in-provider recovery in `makeStreamSimple` has run out of attempts — omp
 * itself will not re-send a turn that already printed text, so the instruction
 * here is genuinely for the user.
 */
function explainTransport(message: string, partial: boolean): string {
	return [
		"Zen's edge dropped the connection before the answer finished — a transport reset on the gateway",
		"side, not the free-tier handshake, the session id, or anything to do with tools.",
		partial
			? "What arrived above is the partial answer and is kept. omp does not replay a turn whose text was already shown, so send any message to have the model continue it."
			: "Nothing was received, so re-running the turn is safe.",
		`(${message.split(" For more information")[0]})`,
	].join(" ")
}

/** Turn the gateway's bare 403 into something a user can act on. */
function explain(message: string): string {
	if (!isHandshakeError(message)) return message
	return [
		"Zen refused the free-tier handshake (403 FreeTierError) on a request that carried the `read` tool.",
		"A stale session id is the usual cause, and this lane mints a fresh one and retries once before",
		"giving up — so a second refusal means the id is not the problem: /ocfree-new-session forces a new",
		"one, and if that still fails the free queue is down for these ids.",
	].join(" ")
}

type StreamSimpleFn = NonNullable<ProviderConfig["streamSimple"]>

/** One turn may be re-sent this many times before anything reached omp. */
const MAX_TRANSPORT_RETRIES = 2
/** Extra requests per turn that continue a partially shown answer. */
const MAX_CONTINUATIONS = 2
/** Chars held back at the start of a continuation to drop a repeated lead. */
const CONTINUATION_HOLD = 480
/** Chars of the partial answer sent back as the assistant turn of a continuation. */
const CONTINUATION_CARRY = 16_000
const CONTINUATION_PROMPT = [
	"Your previous reply was cut off by a network error, mid-answer.",
	"Continue it from exactly the point where it stopped.",
	"Output only the continuation: do not restart, do not repeat anything already written,",
	"and do not mention the interruption.",
].join(" ")

function zeroUsage(): AssistantMessage["usage"] {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	}
}

/**
 * Length of the head of `incoming` that repeats the tail of `emitted`. A resumed
 * model often re-states the last few words before carrying on; that text is
 * already on screen, so it is dropped rather than shown twice.
 */
function repeatedPrefix(emitted: string, incoming: string): number {
	const limit = Math.min(incoming.length, 600, emitted.length)
	for (let size = limit; size >= 16; size--) {
		if (emitted.endsWith(incoming.slice(0, size))) return size
	}
	return 0
}

function makeStreamSimple(): StreamSimpleFn {
	return (model, context, options) => {
		const stream = new AssistantMessageEventStream()
		void (async () => {
			try {
				const key = await resolveZenKey()
				if (!key) {
					throw new Error(
						"No OpenCode Zen key found. Set OPENCODE_API_KEY, or run `opencode auth login` / sign in to opencode-zen in omp.",
					)
				}
				const aborted = () => (options as { signal?: AbortSignal } | undefined)?.signal?.aborted === true
				let sessionId = await ensureSessionId()
				// The gate is not "some tools": measured on the live endpoint, a request
				// passes only when the tool list contains one named `read`. So a turn
				// without it can never succeed, whatever the session id is — which is
				// how `--no-tools` and omp's tool-less effort probe get refused.
				const toolNames = (context.tools ?? []).map(tool => tool.name)
				const hasReadTool = toolNames.includes("read")
				debug(
					`request: model=${model.id} session=${sessionId} tools=${toolNames.length}${hasReadTool ? "" : " (no read tool)"} msgs=${context.messages?.length ?? 0} first=${JSON.stringify(
						String((context.messages?.[0] as { content?: unknown })?.content ?? "").slice(0, 80),
					)}`,
				)

				// `compat` may live as a prototype getter on Model — re-attach it
				// explicitly so the spread does not drop it.
				const compat = {
					...((model as { compat?: Record<string, unknown> }).compat ?? {}),
					...((model as { compatConfig?: Record<string, unknown> }).compatConfig ?? {}),
					supportsUsageInStreaming: true,
					supportsToolChoice: true,
					maxTokensField: "max_tokens",
				}

				const textOf = (blocks: AssistantMessage["content"]): string => {
					let text = ""
					for (const block of blocks) if (block.type === "text") text += block.text
					return text
				}
				/**
				 * What survives an interrupted attempt: only text and thinking. A call
				 * whose argument list was half-written must never be replayed, and a
				 * complete one is either handed over (salvage) or dropped by design.
				 */
				const settled = (blocks: AssistantMessage["content"]): AssistantMessage["content"] =>
					blocks.filter(block => block.type === "text" || block.type === "thinking")

				// A retried attempt replays `start`; the consumer already saw it.
				let started = false
				let retriedStaleId = false
				let transportRetries = 0
				let continuations = 0
				/**
				 * Blocks from interrupted attempts that omp has already been shown. A
				 * continuation carries their text and every later event is re-indexed
				 * behind them, so the whole turn stays one assistant message.
				 */
				let carried: AssistantMessage["content"] = []
				let continuing = false
				let requestContext = context

				for (;;) {
					const headers = zenHeaders(sessionId)
					const pipeModel = {
						...model,
						compat,
						compatConfig: compat,
						api: "openai-completions",
						baseUrl: ZEN_BASE,
						headers,
					}
					// Delegate to pi-ai's own streamer: SSE parsing, native tool calls,
					// reasoning replay, retries — all identical to any other provider.
					const inner = piStreamSimple(pipeModel as never, requestContext as Context, {
						...(options as Record<string, unknown>),
						apiKey: key,
						headers,
					} as SimpleStreamOptions as never)

					// Per-attempt state: `attempt` mirrors exactly what omp has been shown
					// by this request, so no decision depends on untrimmed inner text.
					const emitted = new Map<number, string>()
					let attempt: AssistantMessage["content"] = []
					const rawArgs = new Map<number, string>()
					const callJsonOk = new Map<string, boolean>()
					const pendingCalls = new Set<number>()

					const offset = continuing ? carried.length : 0
					const outIndex = (index: number): number => index + offset
					const snapshot = (partial: AssistantMessage): AssistantMessage => {
						if (!continuing) return partial
						return {
							...partial,
							content: [
								...carried,
								...partial.content.map((block, index) => {
									const shown = block.type === "text" ? emitted.get(index) : undefined
									return shown === undefined ? block : { ...block, text: shown }
								}),
							],
						}
					}

					// `null` once live: only the first block of a continuation is held.
					let hold: string | null = continuing ? "" : null
					let staleId = false
					let retryTransport = false
					let retryContinuation = false
					let finished = false

					const emitText = (index: number, delta: string, partial: AssistantMessage): void => {
						if (!delta) return
						emitted.set(index, (emitted.get(index) ?? "") + delta)
						stream.push({ type: "text_delta", contentIndex: outIndex(index), delta, partial: snapshot(partial) })
					}
					const releaseHold = (index: number, partial: AssistantMessage): void => {
						if (hold === null) return
						const text = hold
						hold = null
						if (!text) return
						const skip = repeatedPrefix(textOf([...carried, ...attempt]), text)
						emitText(index, text.slice(skip), partial)
					}

					attemptLoop: for await (const event of inner) {
						if (event.type === "error") {
							const message = event.error?.errorMessage ?? ""
							const merged: AssistantMessage["content"] = [...carried, ...attempt]
							const partialText = textOf(merged)

							if (isHandshakeError(message) && !hasReadTool) {
								// No session id can rescue a tool-less turn: the gate wants `read`
								// on the wire, so keep the cached id.
								debug(`403 on a turn without the read tool; keeping session ${sessionId}`)
								event.error.errorMessage = [
									"Zen refused this call (403 FreeTierError) because it carries no `read` tool.",
									"The free ids only answer agent turns, so `--no-tools` and omp's tool-less effort",
									"probe cannot use them; the cached session id is untouched.",
								].join(" ")
								stream.push({ ...event, error: { ...event.error, content: merged } })
								finished = true
								break attemptLoop
							}
							if (isHandshakeError(message)) {
								if (!retriedStaleId && merged.length === 0) {
									// A refused agent turn is the one case that says something about
									// the id, so heal it inside this turn instead of asking for a
									// second run.
									debug(`handshake 403 on session ${sessionId}; minting a fresh id and retrying once`)
									forgetOcfreeSession()
									sessionId = await ensureSessionId()
									retriedStaleId = true
									staleId = true
									break attemptLoop
								}
								event.error.errorMessage = explain(message)
								stream.push({ ...event, error: { ...event.error, content: merged } })
								finished = true
								break attemptLoop
							}
							if (!aborted() && isTransportClose(message)) {
								const calls = attempt.filter((block): block is ToolCall => block.type === "toolCall")
								const callsComplete =
									calls.length > 0 &&
									pendingCalls.size === 0 &&
									calls.every(call => callJsonOk.get(call.id) === true)
								if (callsComplete) {
									// The turn's work was already complete on the wire; hand it to
									// omp instead of throwing away a finished tool call.
									debug(`transport close with ${calls.length} complete tool call(s); finishing the turn`)
									stream.push({
										type: "done",
										reason: "toolUse",
										message: {
											...event.error,
											content: merged,
											stopReason: "toolUse",
											errorMessage: undefined,
											errorStatus: undefined,
											errorId: undefined,
											toolCallAbortMessages: undefined,
										},
									})
									finished = true
									break attemptLoop
								}
								if (partialText.trim().length > 0 && continuations < MAX_CONTINUATIONS) {
									// Text is already on screen and omp will not replay this turn, so
									// ask the model to finish its own sentence — bounded, and only
									// the new tail is forwarded.
									carried = settled(merged)
									continuing = true
									continuations++
									debug(
										`transport close after ${partialText.length} chars; continuation ${continuations}/${MAX_CONTINUATIONS}`,
									)
									requestContext = {
										...context,
										messages: [
											...context.messages,
											{
												role: "assistant",
												content: [{ type: "text", text: partialText.slice(-CONTINUATION_CARRY) }],
												api: model.api,
												provider: model.provider,
												model: model.id,
												usage: zeroUsage(),
												stopReason: "stop",
												timestamp: Date.now(),
											},
											{ role: "user", content: CONTINUATION_PROMPT, timestamp: Date.now() },
										],
									}
									retryContinuation = true
									break attemptLoop
								}
								if (
									partialText.trim().length === 0 &&
									attempt.every(block => block.type !== "toolCall") &&
									transportRetries < MAX_TRANSPORT_RETRIES
								) {
									// Nothing reached omp yet, so a fresh attempt is safe — and it
									// recovers both a dropped stream and a dead pooled socket.
									transportRetries++
									debug(`transport close before any content; retry ${transportRetries}/${MAX_TRANSPORT_RETRIES}`)
									retryTransport = true
									break attemptLoop
								}
								event.error.errorMessage = explainTransport(message, partialText.trim().length > 0)
								stream.push({ ...event, error: { ...event.error, content: merged } })
								finished = true
								break attemptLoop
							}
							if (!aborted()) event.error.errorMessage = explain(message)
							stream.push({ ...event, error: { ...event.error, content: merged } })
							finished = true
							break attemptLoop
						}

						if (event.type === "start") {
							if (started) continue
							started = true
							stream.push(event)
							continue
						}
						if (event.type === "done") {
							stream.push(
								continuing
									? { ...event, message: { ...event.message, content: [...carried, ...attempt] } }
									: event,
							)
							finished = true
							break attemptLoop
						}

						switch (event.type) {
							case "text_start":
								stream.push({ ...event, contentIndex: outIndex(event.contentIndex), partial: snapshot(event.partial) })
								break
							case "text_delta": {
								const index = event.contentIndex
								if (hold === null) {
									emitText(index, event.delta, event.partial)
								} else {
									hold += event.delta
									if (hold.length >= CONTINUATION_HOLD) releaseHold(index, event.partial)
								}
								break
							}
							case "text_end": {
								const index = event.contentIndex
								releaseHold(index, event.partial)
								const text = emitted.get(index) ?? ""
								attempt.push({ type: "text", text })
								stream.push({
									...event,
									contentIndex: outIndex(index),
									content: emitted.has(index) ? text : event.content,
									partial: snapshot(event.partial),
								})
								break
							}
							case "thinking_start":
							case "thinking_delta":
								stream.push({ ...event, contentIndex: outIndex(event.contentIndex), partial: snapshot(event.partial) })
								break
							case "thinking_end":
								attempt.push({ type: "thinking", thinking: event.content })
								stream.push({ ...event, contentIndex: outIndex(event.contentIndex), partial: snapshot(event.partial) })
								break
							case "toolcall_start":
								pendingCalls.add(event.contentIndex)
								stream.push({ ...event, contentIndex: outIndex(event.contentIndex), partial: snapshot(event.partial) })
								break
							case "toolcall_delta":
								rawArgs.set(event.contentIndex, (rawArgs.get(event.contentIndex) ?? "") + event.delta)
								stream.push({ ...event, contentIndex: outIndex(event.contentIndex), partial: snapshot(event.partial) })
								break
							case "toolcall_end": {
								const index = event.contentIndex
								pendingCalls.delete(index)
								const raw = rawArgs.get(index) ?? ""
								let ok = true
								try {
									JSON.parse(raw.trim() === "" ? "{}" : raw)
								} catch {
									ok = false
								}
								callJsonOk.set(event.toolCall.id, ok)
								attempt.push(event.toolCall)
								stream.push({ ...event, contentIndex: outIndex(index), partial: snapshot(event.partial) })
								break
							}
							case "image_end":
								attempt.push(event.content)
								stream.push({ ...event, contentIndex: outIndex(event.contentIndex), partial: snapshot(event.partial) })
								break
						}
					}

					if (finished) break
					if (staleId) continue
					if (retryContinuation) {
						await sleep(300)
						continue
					}
					if (retryTransport) {
						await sleep(500 * transportRetries)
						continue
					}
					break
				}
				stream.end()
			} catch (error) {
				const message = explain(error instanceof Error ? error.message : String(error))
				stream.push({
					type: "error",
					reason: "error",
					error: {
						role: "assistant",
						content: [],
						api: API,
						provider: PROVIDER,
						model: model.id,
						usage: zeroUsage(),
						stopReason: "error",
						errorMessage: message,
						timestamp: Date.now(),
					},
				} as never)
				stream.end()
			}
		})()
		return stream
	}
}
// ── Extension entry ──────────────────────────────────────────────────────
export function registerOcfreeProvider(pi: ExtensionAPI): void {
	pi.registerProvider(PROVIDER, {
		baseUrl: ZEN_BASE,
		api: API,
		streamSimple: makeStreamSimple(),
		authHeader: false,
		// The free ids are a fixed catalog on Zen's side, so this answers from the
		// constant above — omp still runs it through its usual model cache.
		fetchDynamicModels: async () => FREE_MODELS,
	} as unknown as ProviderConfig)

	pi.registerCommand("ocfree-new-session", {
		description: "Forget the cached Zen session id; the next request mints a fresh one",
		handler: async (_args, ctx) => {
			forgetOcfreeSession()
			ctx.ui.notify("ocfree: cached session id cleared — the next request mints a new one.", "info")
		},
	})

	pi.registerCommand("ocfree-status", {
		description: "Show the ocfree lane: endpoint, cached session, and the free model ids",
		handler: async (_args, ctx) => {
			const session = readCachedSession()
			const lines = [
				`ocfree: ${FREE_MODELS.length} free Zen models, native tool calling`,
				`endpoint: ${ZEN_BASE} (direct, no running opencode)`,
				session.id && isValidSessionId(session.id)
					? `session: ${session.id} (minted ${session.mintedAt ? new Date(session.mintedAt).toLocaleString() : "earlier"})`
					: "session: none cached — it is minted on the first request",
				`ids: ${FREE_MODELS.map(m => m.id).join(", ")}`,
			]
			ctx.ui.notify(lines.join("\n"), "info")
		},
	})
}
