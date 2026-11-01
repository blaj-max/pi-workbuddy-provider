# pi-workbuddy-provider

A provider plugin for [oh-my-pi](https://github.com/oh-my-pi) (`omp`) that serves the free models on WorkBuddy / CodeBuddy's own gateway, straight to omp — with native streaming, tool calls, reasoning and images.

It talks to `https://www.codebuddy.ai/v2/chat/completions` (or the `.cn` / `workbuddy.ai` hosts) using the session you sign in with. The plugin does not bundle the desktop app, and no longer needs it running: browser sign-in writes to the plugin's own credential store.

## Read this first

WorkBuddy has started rejecting non-official clients. The gateway returns `403 code 11140` — "your account has been restricted due to unusual activity" — for accounts it considers out of policy, and that applies to the official app and CLI on the same account too. **Using this plugin can get the account you sign in with restricted.** This is not a bug that can be patched around; it is the upstream's enforcement.

- Use an account you are willing to lose access to.
- The desktop app is always the lower-risk client.
- This project is not affiliated with, endorsed by, or supported by Tencent or WorkBuddy.
- The gateway is unofficial and can change without notice.

If you are already restricted, the only paths are the upstream's appeal flow (the app's feedback action) or a different account. The account pool in this plugin exists so a second account can take over without editing files.

## Requirements

- omp `v18` or newer, Node `22+` on Linux/macOS (Windows works for the HTTP path; the desktop credential discovery is Linux/macOS first).
- Nothing else for the browser sign-in. The bundled Zen provider (`opencode Zen` free models) mints its session through the `opencode` CLI once, so that half needs `npm i -g opencode-ai` unless `OCFREE_OPENCODE_BIN` points at a binary.

## Install

```bash
omp plugin install pi-workbuddy-provider
```

Then start omp. The `workbuddy` provider appears in the model picker with the models that are free *right now* (base price `x0.00` or an active factor-0 promotion), and a `zen` provider for the OpenCode Zen free ids.

To install from a checkout instead:

```bash
omp plugin install /path/to/pi-workbuddy-provider
```

## Sign in (no desktop app)

```
/workbuddy-login              # international, workbuddy.ai
/workbuddy-login codebuddy    # international, codebuddy.ai brand
/workbuddy-login cn           # China, codebuddy.cn
/workbuddy-login https://…    # any compatible base
```

The command opens the authorization page and waits up to ten minutes. When the browser flow finishes, the tokens are stored in `~/.omp/workbuddy-vault.json` (mode `0600`) and that account becomes active. The process never needs the desktop app, and the app's own login file is only read when it exists.

If WorkBuddy is already signed in on this machine, that session is picked up automatically as a fallback account — even before you run `/workbuddy-login`.

## Accounts and failover

```
/workbuddy-accounts
```

lists stored accounts and lets you pick the one requests try first. Each request starts on the active account and, if the upstream says the *account* is the problem, moves down the list:

| failure | behavior |
|---|---|
| `401` (token rejected), `402` (no credits), `429` (rate limit) | next account |
| `403` with `11140` / "restricted" | next account, cooled down for 24 h |
| `502` / `503` / `504` | next account |
| `400`, or any failure after content has streamed | reported as-is — replaying would duplicate the turn |

A failed account is cooled down locally for the window the gateway names (or 15 min for a bare 429, 6 h for no credits). Cooldowns are in memory: restarting omp tries everyone again. The warning written to omp's log names which account failed and which one takes over.

Only the *active* account's token is used for catalog reads (`/v3/config`), so promoted/free models always match the account that will serve the turn.

## Commands

| command | effect |
|---|---|
| `/workbuddy-login [cn\|codebuddy\|global\|URL]` | browser sign-in; adds an account and makes it active |
| `/workbuddy-accounts` | list accounts, switch the active one |
| `/workbuddy-refresh` | re-pull the free-model list (promotions rotate server-side) |

## What it does

- **Direct pipe.** `streamSimple` delegates to pi-ai's own `openai-completions` streamer, so tool calls, reasoning replay, retries and SSE parsing are not reimplemented here.
- **Client identity.** Requests carry the desktop client's headers (`X-IDE-Name: WorkBuddy`, `X-Product-Version`, conversation/trace ids per request) because the gateway rejects the bare OpenAI shape.
- **Free models only.** `/v3/config` is filtered to models whose base price is `x0.00` or that an active factor-0 promotion covers. A saved list stands in for up to 24 h when the catalog is temporarily unreachable, and says so in the log.
- **Thinking surface.** Each model's declared efforts (`low`/`high`/`max`, or the legacy single level) become omp's effort picker; `requiresEffort` is set only when the catalog explicitly says thinking cannot be turned off.
- **Context budget.** The catalog's own `contextWindow.defaultLength` is used when published, not the larger validation ceiling — the gateway's upstream clock rejects very large cold-cache prompts with a 500 before that ceiling matters.
- **`max_tokens` always sent.** Without it the gateway reserves an implicit output budget and rejects long prompts with `400 11133`.
- **DNS blackhole handling.** `www.codebuddy.ai` sometimes resolves to `0.0.0.1`; the plugin follows the zone's own CNAME to the edge that still serves it while `Host`/TLS stay on the canonical name.
- **Readable errors.** Reset timestamps in 429 messages get a local-time rendering appended; the original text and `retry-after` hints are preserved.
- **Zen free ids.** A second provider (`zen`) points at OpenCode Zen's OpenAI-compatible endpoint; the free-tier session id is minted once through `opencode serve` and cached in `~/.omp/cache/ocfree/`.

## Configuration

| variable | meaning |
|---|---|
| `WORKBUDDY_VAULT` | credential file (default `~/.omp/workbuddy-vault.json`) |
| `WORKBUDDY_AUTH_FILE` | explicit desktop login file, instead of discovery |
| `WORKBUDDY_AUTH_DIR` | directory to scan for `*.info` login files |
| `CODEBUDDY_ENDPOINT` | override the API base (testing, or a mirror) |
| `OCFREE_OPENCODE_BIN` | `opencode` binary for the Zen session mint |
| `OPENCODE_API_KEY` | Zen key; otherwise omp's stored `opencode-zen` credential is used |

Free-model discovery is refreshed on omp startup with a 24 h cache. omp stores the resulting list in `~/.omp/agent/models.db`; if a plugin change is not visible after a restart, delete the `workbuddy` row from that table and start omp once.

## Uninstall

```bash
omp plugin uninstall pi-workbuddy-provider
```

Removing the vault file (`~/.omp/workbuddy-vault.json`) revokes nothing upstream — it only forgets the tokens on this machine. To actually end a session, sign out in WorkBuddy.

## License

MIT — see [LICENSE](LICENSE).
