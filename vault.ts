/**
 * Credential store for the workbuddy provider.
 *
 * The desktop app's login file is one *source* of an account, not the only
 * one: browser sign-in (`oauth.ts`) writes here too, and a refresh lands here
 * instead of in the app's file. Keeping the tokens in one store is what makes
 * an account pool possible — a request can move to the next account when the
 * current one is rate-limited, out of credits or restricted, without needing a
 * second desktop install.
 *
 * File: `$WORKBUDDY_VAULT` or `~/.omp/workbuddy-vault.json`, mode 0600.
 * Tokens never leave the process except back to the gateway they came from.
 */
import { createHash } from "node:crypto"
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { homedir } from "node:os"

export type AccountSource = "oauth" | "desktop"

export interface StoredAccount {
	/** Stable id: hash of uid + domain, so re-importing the same login updates one row. */
	id: string
	uid: string
	nickname?: string
	domain: string
	endpoint: string
	access: string
	refresh: string
	/** Access-token expiry, epoch ms (0 = upstream did not say). */
	expiresAt: number
	refreshExpiresAt?: number
	source: AccountSource
	addedAt: number
}

interface VaultFile {
	version: 1
	active?: string
	accounts: StoredAccount[]
}

const EMPTY: VaultFile = { version: 1, accounts: [] }

export function vaultPath(): string {
	return process.env.WORKBUDDY_VAULT || join(homedir(), ".omp", "workbuddy-vault.json")
}

/**
 * A deterministic id for a login. The same account discovered from the desktop
 * file and later signed in through the browser is one account, not two.
 */
export function accountId(uid: string, domain: string): string {
	const digest = createHash("sha256").update(`${uid}|${domain}`).digest("hex")
	return `wb_${digest.slice(0, 12)}`
}

/** Read the vault; a missing or unreadable file is an empty vault, never an error. */
export function readVault(): VaultFile {
	const path = vaultPath()
	if (!existsSync(path)) return { ...EMPTY, accounts: [] }
	try {
		const raw = JSON.parse(readFileSync(path, "utf8")) as Partial<VaultFile>
		const accounts = Array.isArray(raw.accounts)
			? raw.accounts.filter(
					(entry): entry is StoredAccount =>
						typeof entry === "object" &&
						entry !== null &&
						typeof (entry as StoredAccount).id === "string" &&
						typeof (entry as StoredAccount).access === "string" &&
						typeof (entry as StoredAccount).endpoint === "string",
				)
			: []
		return {
			version: 1,
			...(typeof raw.active === "string" ? { active: raw.active } : {}),
			accounts,
		}
	} catch {
		return { ...EMPTY, accounts: [] }
	}
}

function writeVault(vault: VaultFile): void {
	const path = vaultPath()
	mkdirSync(dirname(path), { recursive: true })
	const tmp = `${path}.tmp`
	writeFileSync(tmp, `${JSON.stringify(vault, null, 2)}\n`, { mode: 0o600 })
	chmodSync(tmp, 0o600)
	renameSync(tmp, path)
}

/** Every stored account, in the order the user last used them first. */
export function storedAccounts(): StoredAccount[] {
	const vault = readVault()
	const active = vault.active
	return [...vault.accounts].sort((a, b) => {
		if (a.id === active) return -1
		if (b.id === active) return 1
		return b.addedAt - a.addedAt
	})
}

/** The id of the account requests should try first, if the user picked one. */
export function activeAccountId(): string | undefined {
	return readVault().active
}

/** Make `id` the first account requests use. Returns false when it is unknown. */
export function setActiveAccount(id: string): boolean {
	const vault = readVault()
	if (!vault.accounts.some(account => account.id === id)) return false
	vault.active = id
	writeVault(vault)
	return true
}

/**
 * Insert or update one account.
 *
 * Tokens are the part that changes: an existing row keeps its `addedAt` so the
 * pool's order does not jump every time a token is refreshed. `makeActive`
 * is used by sign-in (the account you just added is the one you want to use)
 * but never by a refresh.
 */
export function upsertAccount(account: StoredAccount, options?: { makeActive?: boolean }): StoredAccount {
	const vault = readVault()
	const index = vault.accounts.findIndex(entry => entry.id === account.id)
	const next = index >= 0 ? { ...vault.accounts[index], ...account, addedAt: vault.accounts[index].addedAt } : account
	if (index >= 0) vault.accounts[index] = next
	else vault.accounts.push(next)
	if (options?.makeActive || vault.accounts.length === 1) vault.active = next.id
	writeVault(vault)
	return next
}

/** Persist refreshed tokens without changing the pool order or the active pick. */
export function updateTokens(
	id: string,
	tokens: { access: string; refresh?: string; expiresAt: number; refreshExpiresAt?: number },
): void {
	const vault = readVault()
	const account = vault.accounts.find(entry => entry.id === id)
	if (!account) return
	account.access = tokens.access
	if (tokens.refresh) account.refresh = tokens.refresh
	account.expiresAt = tokens.expiresAt
	if (tokens.refreshExpiresAt) account.refreshExpiresAt = tokens.refreshExpiresAt
	writeVault(vault)
}

/** Remove one account. The desktop file itself is never touched. */
export function removeAccount(id: string): boolean {
	const vault = readVault()
	const before = vault.accounts.length
	vault.accounts = vault.accounts.filter(account => account.id !== id)
	if (vault.accounts.length === before) return false
	if (vault.active === id) vault.active = vault.accounts[0]?.id
	writeVault(vault)
	return true
}
