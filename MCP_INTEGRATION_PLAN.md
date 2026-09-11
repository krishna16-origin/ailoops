# MCP Integration Plan for ailoops

## Executive summary

The repository is a lightweight FastAPI application with a single-page frontend. It currently supports two modes—Chat and Code—but they do not share a true tool-execution layer:

- **Chat mode** uses a backend heuristic (`needs_web_search`) to decide whether to call Tavily before the model response. The model is explicitly told that it cannot call tools.
- **Code mode** uses a custom text protocol (`THOUGHT`, `ACTION`, `PATH`, fenced content) parsed in `app.py`. Its actions are dispatched directly to the E2B sandbox and include file operations, shell commands, tests, archive handling, and web search.
- **The sidebar** currently contains session history and small settings panels. It is a natural location for an “MCP integrations” entry point.
- **The frontend** is a single `frontend/index.html` file with inline CSS and JavaScript. Streaming is handled through SSE and already has a rich event vocabulary for status, activity, file diffs, terminal output, and completion.
- **State is in memory** (`sessions` in `app.py` and `_sessions` in `sandbox_manager.py`), so connector configuration and permissions must initially be treated as process-local unless persistence is added deliberately.

The recommended implementation is a shared **MCP Gateway** inside the FastAPI backend. It should own server registration, process lifecycle, transport handling, tool discovery, per-session permissions, timeout/cancellation, normalized tool calls, and audit events. Both Chat and Code should call that gateway through the same internal interface and stream the same tool events to the existing SSE UI.

## Repository findings

| Area | Current implementation | Consequence for MCP work |
|---|---|---|
| Backend | `app.py` contains model routing, web search, prompt construction, agent loops, SSE endpoints, and Code action dispatch | Split MCP code into modules before adding integrations; avoid growing `app.py` further |
| Chat tools | Tavily is pre-called by keyword heuristics around `needs_web_search()` and `web_search()` | Replace or supplement this with model-visible tools selected from the enabled MCP registry |
| Code tools | `parse_agent_turn()` and `_run_agent()` implement a text-based action protocol | Add a structured tool path; retain the legacy parser temporarily for compatibility and fallback |
| Streaming | Chat and Code both emit SSE, but with slightly different event handling | Define one normalized event contract for `tool_call`, `tool_result`, `tool_error`, and approval-required states |
| Code execution | `sandbox_manager.py` owns E2B sessions, file synchronization, commands, tests, and preview servers | Filesystem/Git/SQLite MCPs must be scoped carefully so they do not bypass the E2B project boundary |
| Frontend | One large `frontend/index.html`; mode switch is `switchMode('chat'|'code')` | Build reusable MCP drawer/modal components and a shared tool activity renderer in this file first; extract later if needed |
| Dependencies | Python dependencies are listed in `requirements.txt`; no MCP SDK is present | Add a pinned MCP SDK and transport dependencies, with a compatibility test before rollout |
| Persistence | Sessions and sandbox registry are in memory | Start with process-local configuration for development; add durable storage/auth before multi-user deployment |

## Target architecture

```text
                         +-----------------------------+
                         |        frontend/index.html  |
                         | Chat + Code + sidebar       |
                         | MCP drawer + tool activity  |
                         +--------------+--------------+
                                        |
                              SSE / JSON HTTP API
                                        |
                         +--------------v--------------+
                         |          FastAPI app         |
                         |                              |
                         |  Chat orchestration          |
                         |  Code orchestration          |
                         |  Shared MCP tool broker      |
                         +--------------+--------------+
                                        |
                         +--------------v--------------+
                         |          MCP Gateway         |
                         |                              |
                         | Registry / discovery         |
                         | Client sessions / transports |
                         | Permissions / approvals      |
                         | Timeouts / cancellation      |
                         | Audit + normalized events    |
                         +--------------+--------------+
                                        |
       +-------------+-------------+----+----+-------------+-------------+
       |             |             |         |             |             |
  stdio MCP     HTTP MCP       local     remote      SQLite      public APIs
  processes     endpoints      servers   servers     boundary    weather/maps/etc.
```

### Core backend modules to add

Create a package such as `mcp_gateway/` rather than implementing MCP logic directly in `app.py`:

- `mcp_gateway/models.py`: connector/server definitions, tool metadata, invocation requests/results, policy and audit models.
- `mcp_gateway/registry.py`: built-in server catalog plus user-defined server CRUD and enable/disable state.
- `mcp_gateway/client.py`: MCP SDK client creation, stdio/HTTP transport support, initialization, capability negotiation, and tool discovery.
- `mcp_gateway/manager.py`: process lifecycle, connection pooling, reconnects, per-session handles, and graceful shutdown.
- `mcp_gateway/policy.py`: allowlists, sensitive-tool classification, path restrictions, confirmation policy, and rate limits.
- `mcp_gateway/broker.py`: one `list_tools()` and one `call_tool()` interface used by both modes.
- `mcp_gateway/events.py`: normalized event types mapped to existing SSE.
- `mcp_gateway/builtins.py`: metadata and defaults for the ten requested servers.
- `mcp_gateway/storage.py`: initially an in-memory adapter; later a durable database adapter.

The internal contract should look conceptually like:

```python
available_tools = await mcp_broker.list_tools(
    session_id=session_id,
    mode="chat" | "code",
    enabled_servers=...
)

result = await mcp_broker.call_tool(
    session_id=session_id,
    server_id="open-meteo",
    tool_name="...",
    arguments={...},
    mode=mode,
    emit=emit,
)
```

The broker must return structured, size-limited results. It should never expose raw credentials, unrestricted host paths, or arbitrary server process arguments to the model.

## Requested MCP rollout

The ten integrations should not all be enabled in the first release. Group them by risk and runtime complexity:

| Group | MCPs | Initial treatment |
|---|---|---|
| Local development primitives | Filesystem, Git, SQLite | Implement behind strict workspace and repository policies; highest security review priority |
| Browser/network capability | Fetch, Playwright | Add explicit network/domain policy, timeouts, and user-visible activity; Playwright should be opt-in because it can interact with authenticated sites |
| Memory | Memory | Add namespaced, user/session-scoped storage and clear/delete controls before enabling by default |
| Public read-only knowledge/data | Open-Meteo, Nominatim/OpenStreetMap, Wikipedia/Wikidata, arXiv | Prefer official/public endpoints or maintained MCP adapters; add attribution, rate limiting, caching, and source links |

### 1. Filesystem MCP

- Scope it to the current Code project directory, not the host filesystem.
- For Code mode, map the MCP filesystem root to the E2B project workspace or a controlled virtual workspace.
- For Chat mode, default to read-only and do not expose arbitrary local files unless the user explicitly attaches or authorizes a workspace.
- Classify delete, move, overwrite, and recursive operations as confirmation-required.
- Add path normalization and traversal prevention tests (`..`, symlinks, absolute paths, hidden files, and generated artifacts).

### 2. Fetch MCP

- Provide controlled HTTP retrieval for Chat and Code.
- Enforce URL scheme restrictions, redirect limits, response-size limits, content-type checks, and request timeouts.
- Add SSRF protection: deny loopback, link-local, metadata-service, private-network, and local Unix/socket targets unless explicitly allowed for a development session.
- Normalize results into text plus URL/title/content metadata so the existing source rendering can show citations.

### 3. Playwright MCP

- Run in an isolated browser process/container with a bounded lifetime.
- Make browser actions visible in the existing Code “Watch live” activity panel and in Chat as tool activity.
- Default to read-only navigation. Require confirmation for form submission, purchases, account/security changes, public posting, downloads, and authenticated actions.
- Store no cookies or browser profile data by default; make persistence an explicit setting.
- Add action limits, domain allowlists, popup/download policies, and screenshot/HTML size limits.

### 4. Git MCP

- Scope repositories to the controlled project workspace.
- Support status, diff, log, branch inspection, and safe read operations first.
- Treat commit, branch creation, merge, push, reset, checkout, and remote changes as separate capabilities with explicit policy checks.
- Never pass credentials or remote URLs containing secrets into model-visible output.
- Integrate tool results with Code’s file/diff view rather than duplicating a second repository UI.

### 5. SQLite MCP

- Start with a per-session or per-project database file inside the controlled workspace.
- Enforce read-only mode by default for Chat; Code may write only to explicitly selected project databases.
- Add query timeout, row/column/result-size limits, and prohibit unsafe extensions or arbitrary file attachment paths.
- Add a schema/result renderer and CSV/JSON export only after size controls are in place.

### 6. Memory MCP

- Define namespaces: user, project, conversation, and ephemeral run.
- Add metadata and retention fields so memory can be listed, edited, deleted, and cleared.
- Require user-facing controls in the MCP sidebar; memory must not be an invisible side effect.
- Do not store secrets, full credentials, browser cookies, or unrestricted file contents.
- Include memory provenance in retrieval results so the model can distinguish remembered preferences from current facts.

### 7. Open-Meteo

- Expose geocoding and forecast tools through a weather adapter/MCP server.
- Resolve location before forecast calls and preserve units/timezone in the result.
- Cache short-lived weather responses and show the observation/forecast timestamp.
- Treat it as read-only and low risk, but still apply rate limits and source attribution.

### 8. OpenStreetMap / Nominatim

- Separate geocoding/search from map or place details in the tool metadata.
- Respect the public service usage policy, identify the application, rate-limit requests, and cache responsibly.
- Return attribution and coordinates in a stable structure.
- Add an explicit “near me” privacy rule: never infer or transmit precise user location without an explicit location input or permission.

### 9. Wikipedia / Wikidata

- Provide search, page retrieval, entity lookup, and structured fact retrieval as separate read-only tools.
- Preserve page/entity IDs and source URLs for citations.
- Handle disambiguation and stale/contradictory statements explicitly; do not present Wikidata values as current without a date/qualifier check.
- Add result truncation and language selection.

### 10. arXiv

- Provide search, metadata retrieval, abstract retrieval, and optionally PDF download/extraction as separate tools.
- Include arXiv IDs, version numbers, authors, dates, and source URLs in results.
- Rate-limit searches, cap result counts, and avoid silently downloading large PDFs.
- If PDF extraction is added, run it in the existing isolated Code sandbox or a separate bounded worker and return citations, not unbounded document text.

## Shared Chat and Code behavior

### Chat mode

1. The client sends the user message plus selected MCP server IDs and per-turn permissions.
2. The backend discovers only the tools allowed for that turn and includes compact tool schemas in the model request.
3. The model can request a tool through the broker using structured arguments.
4. The broker validates policy, optionally emits `tool_approval_required`, invokes the MCP server, truncates/normalizes the result, and emits `tool_result`.
5. The model receives the result and continues until it returns a final answer or reaches a tool/turn budget.
6. The final answer includes source links or structured citations when the tool result supports them.

The existing keyword-driven Tavily path can remain as a temporary fallback, but it should no longer be the long-term routing mechanism. The prompt text around lines 633–652 of `app.py` must be updated because it currently says the model has no live tool to call.

### Code mode

1. The Code agent gets the same broker tool catalog, plus project-scoped Filesystem/Git/SQLite tools and the existing E2B operations.
2. Structured MCP calls are preferred for reads, repository inspection, database queries, web retrieval, and browser work.
3. Existing `THOUGHT/ACTION/PATH` parsing remains as a compatibility fallback during migration, but new MCP operations must not be encoded as free-form text actions.
4. MCP activity events appear beside existing file edits, terminal output, plan steps, and sandbox status in “Watch live”.
5. Tool outputs are added to the Code agent context with clear provenance and bounded size; file changes still flow through the existing `code_files`/sandbox synchronization path.

## Frontend/sidebar plan

Add an **MCP integrations** row in the existing sidebar below history and above the mode settings. Clicking it opens a drawer or modal with:

- Enabled/disabled status for each server.
- Server category and short capability summary.
- Connection status: disconnected, connecting, ready, error.
- Tool count and last-used timestamp.
- Per-mode toggles: available in Chat, available in Code.
- Per-turn “use MCPs” control in the composer, defaulting to the user’s saved preference.
- Add custom MCP button for URL or local command configuration.
- Test connection and refresh tools actions.
- Permission controls: read-only, ask before writes, allow writes.
- Remove/disable controls for user-created servers.
- Memory management controls for the Memory MCP.

Use one visual language for both modes. The existing SSE activity renderer should gain rows for connecting, calling, succeeded, blocked, awaiting approval, and failed. Do not create a second tool timeline for Chat; reuse the same component and event schema.

## Custom MCP plugin flow

The “Add custom MCP” form should support two installation types:

1. **Remote MCP URL**: URL, display name, optional non-secret metadata, transport type, and auth configuration.
2. **Local stdio command**: command, arguments, environment variable names/values, working directory, and an explicit workspace restriction.

Security requirements:

- Secrets must be stored server-side and masked in the UI; never place them in model prompts, URLs, logs, SSE events, or tool results.
- Validate and normalize commands/arguments; do not accept shell interpolation.
- Require a capability review after discovery: show server name, tool names, descriptions, and requested capabilities before enabling.
- Default custom servers to disabled and read-only until the user enables them.
- Add a kill/disconnect path and a maximum process count per user/session.
- Log connector lifecycle and tool invocations without sensitive arguments.
- Do not allow a custom server to escape the app’s workspace policy merely because it declares a filesystem tool.

## API surface to add

Proposed backend endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /mcp/servers` | List built-in and user-configured servers with redacted status |
| `POST /mcp/servers` | Create a custom server configuration; validate but do not expose secrets |
| `PATCH /mcp/servers/{id}` | Enable/disable or update policy/configuration |
| `DELETE /mcp/servers/{id}` | Remove a user-created server after explicit UI action |
| `POST /mcp/servers/{id}/connect` | Start connection and discover capabilities |
| `POST /mcp/servers/{id}/disconnect` | Stop connection/process |
| `GET /mcp/servers/{id}/tools` | Return discovered, redacted tool metadata |
| `POST /mcp/servers/{id}/test` | Execute a safe health/discovery test |
| `GET /mcp/policies` | Return available policy controls and defaults |
| `PATCH /mcp/policies` | Update user/project policies |

The existing `/chat` and `/code-chat` request models should gain an `mcp` object containing selected server IDs, mode enablement, and per-turn approval settings. Keep the field optional so existing clients remain compatible.

## Event contract

Add the following common SSE events to both streams:

```json
{"type":"mcp_status","server_id":"open-meteo","state":"ready","tool_count":4}
{"type":"tool_call","call_id":"...","server_id":"open-meteo","tool_name":"...","arguments_preview":"..."}
{"type":"tool_approval_required","call_id":"...","reason":"This action writes to the project"}
{"type":"tool_result","call_id":"...","server_id":"open-meteo","tool_name":"...","result":"...","sources":[]}
{"type":"tool_error","call_id":"...","server_id":"...","message":"..."}
```

Arguments and results should be redacted/truncated before emission. The browser should render the same event types regardless of Chat or Code mode.

## Delivery phases

### Phase 0: foundation and threat model

- Add MCP SDK dependency and confirm supported transports.
- Create the `mcp_gateway` package and typed models.
- Add an in-memory registry and fake MCP server for tests.
- Define path, network, process, data-retention, and approval policies.
- Add unit tests for redaction, limits, cancellation, traversal prevention, SSRF blocking, and session isolation.

**Exit criteria:** fake server can be discovered and invoked through the broker; no existing Chat/Code behavior changes.

### Phase 1: shared UI and observability

- Add sidebar MCP entry and drawer/modal.
- Add server status, enablement, connection test, and tool discovery screens.
- Add shared SSE event rendering in Chat and Code.
- Add request fields and session preference handling.

**Exit criteria:** a fake server can be enabled from the sidebar and its test call appears identically in both modes.

### Phase 2: low-risk read-only public servers

- Add Open-Meteo, Wikipedia/Wikidata, arXiv, and controlled Fetch.
- Add citations/source rendering and response limits.
- Keep legacy Tavily search as fallback during comparison testing.

**Exit criteria:** representative weather, entity, article, paper, and URL retrieval tasks work in both modes with citations and rate limits.

### Phase 3: local developer servers

- Add Filesystem, Git, and SQLite with workspace/session scoping.
- Integrate results with Code’s file store, diffs, sandbox, and project preview.
- Add write/delete/commit/query approval flows.

**Exit criteria:** Code can inspect and modify only the authorized project; unsafe paths and destructive operations are blocked or require approval.

### Phase 4: browser and memory

- Add Playwright in an isolated runtime with domain/action policies.
- Add Memory with namespaces, retention, provenance, and user controls.
- Add approval UI for authenticated or consequential browser actions.

**Exit criteria:** browser and memory actions are visible, bounded, auditable, and removable by the user.

### Phase 5: custom plugin ecosystem

- Add remote URL and local stdio custom-server configuration.
- Add secret storage, capability review, lifecycle controls, and per-user quotas.
- Add persistence and authentication if the application is deployed for multiple users.

**Exit criteria:** a user can safely add, inspect, test, enable, disable, and remove their own MCP without server restart or source-code changes.

## Testing and verification strategy

### Unit tests

- Registry CRUD and duplicate-server handling.
- Tool schema normalization and unsupported-schema behavior.
- Policy checks for paths, URLs, commands, SQL, browser actions, and writes.
- Secret redaction in logs, events, errors, and model context.
- Result truncation and binary/document handling.
- Timeout, cancellation, retry, disconnect, and reconnect behavior.
- Session/user/project isolation.

### Integration tests

- Start a fake stdio MCP server and a fake HTTP MCP server.
- Discover tools, invoke them, stream events, and shut them down.
- Run the same tool task through `/chat` and `/code-chat`.
- Verify legacy Chat search and Code parser still work when MCPs are disabled.
- Verify a failed MCP does not terminate the model stream or corrupt session state.

### Security tests

- Filesystem traversal and symlink escape.
- SSRF and redirect abuse in Fetch/Playwright.
- Shell injection through custom stdio configuration.
- SQL injection is not the primary concern for SQLite, but unsafe attachment paths/extensions must be blocked.
- Credential leakage through tool results and browser pages.
- Prompt injection in fetched pages, repositories, Wikipedia pages, papers, and browser content. Treat all tool output as untrusted data and preserve the system policy boundary.

### UX checks

- MCP configuration is discoverable in the sidebar in both modes.
- Tool activity is understandable without exposing hidden model reasoning.
- Long-running tools show progress/heartbeat and can be cancelled.
- Approval prompts show the exact action and material target before execution.
- Mobile/narrow layout does not hide the MCP drawer or composer controls.

## Recommended first implementation slice

Build the smallest vertical slice before implementing all ten servers:

1. Add the gateway, fake server, typed call/result models, and common SSE events.
2. Add the sidebar MCP drawer and a per-turn MCP toggle.
3. Add one read-only public adapter, preferably Open-Meteo, and one local adapter, preferably Filesystem scoped to the E2B project.
4. Wire both adapters into Chat and Code using the same broker.
5. Verify tool discovery, invocation, errors, cancellation, event rendering, and permission enforcement.
6. Only then add the remaining servers in the phased order above.

This sequence validates the hardest product requirement—**the same MCP experience in Chat and Code with one sidebar management UI**—before investing in the full connector catalog.

## Key risks and decisions

| Risk | Mitigation |
|---|---|
| MCP server can execute arbitrary local actions | Workspace isolation, capability policies, process limits, and explicit approval for writes |
| Model follows prompt injection from fetched/browser/repository content | Mark tool output as untrusted data; never treat it as system/developer instructions |
| Current text protocol conflicts with structured tool calls | Add structured MCP path first and retain parser fallback during migration |
| Single-file frontend becomes unmaintainable | Implement the first slice in existing file, then extract components/modules after behavior stabilizes |
| In-memory state is lost on restart and unsafe for multi-user use | Start local-only; add authenticated persistent connector storage before production multi-user deployment |
| Public services throttle or change policies | Per-service adapters, attribution, caching, rate limits, health status, and configurable endpoints |
| Long-running tools stall SSE | Heartbeats, cancellation, per-tool deadlines, and bounded concurrency |
| Tool catalog overwhelms model context | Filter by enabled server/mode/task, cap schemas, and group tools by server |

## Final recommendation

Implement the **shared MCP Gateway + sidebar management UI + common SSE event contract** first. Treat MCP servers as declarative capabilities behind policy, not as direct additions to the current Code action parser. Roll out read-only public data sources before local mutation tools, and keep Filesystem/Git/SQLite/Playwright behind explicit workspace, network, and approval controls. This preserves the current Chat/Code experience while creating one extensible plugin path for the ten requested MCPs and future user-added servers.
