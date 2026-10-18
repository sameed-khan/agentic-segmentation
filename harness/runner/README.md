# pi SDK batch runner

Batch-executes the MSTT agent harness: one pi `AgentSession` per case, a shared
`ModelRuntime`, a concurrency pool, and one output directory per case. Design:
`research/harness_design_v2.md` §8.3; HTTP contract: `harness/server/CONTRACTS.md` §5–6.

## Install

```bash
cd harness/runner && npm install        # pins @earendil-works/* exactly to 1.2.0
cd ../extension && npm install
```

Nothing is installed globally. Node ≥ 22; TypeScript is run with `tsx`, so there
is no build step. `@earendil-works/pi-coding-agent` lives directly in this
instance of the runner, so `pi update` on the machine cannot change the
experiment. No `models.json` is needed: the built-in model catalog plus the
user's `~/.pi/agent/auth.json` are used in place (OAuth refreshes write back to
the same file).

## Run

```bash
cd harness/runner
npx tsx run_cases.ts --split design --run-id phase1-01 --concurrency 4
npx tsx run_cases.ts --cases F14,M07 --run-id try-01
```

Options (`npx tsx run_cases.ts --help`):

| Option | Default | Meaning |
|---|---|---|
| `--cases F14,M07` | – | explicit case ids (from `eval/v2/mstt_cases.csv`) |
| `--split design\|test\|all` | – | every case in that split |
| `--run-id <id>` | timestamp | output directory under `results/` |
| `--concurrency <n>` | 4 | cases in flight in this one process |
| `--model <provider/id>` | `anthropic/claude-sonnet-5-5` | |
| `--thinking <level>` | `medium` | |
| `--wall-min <n>` | 10 | per-case wall-time limit (`session.abort()`) |
| `--max-nudges <n>` | 3 | follow-up "call `seg_finish`" messages |
| `--max-tool-calls <n>` | 40 | sent to segd and substituted into the system prompt |
| `--max-model-prompts <n>` | 15 | sent to segd and substituted into the system prompt |
| `--segd-url <url>` | `http://127.0.0.1:8765` | |
| `--nni-url <url>` | `http://127.0.0.1:1527` | |
| `--extension <path>` | `harness/extension` | override for smoke tests (`harness/checks/ping_ext.ts`) |
| `--dry-stub` | off | spawn `test/stub_segd.mjs` and use it as segd |
| `--auth-shim <path>` | `~/.pi/agent/npm/node_modules/@gotgenes/pi-anthropic-auth` | |
| `--results-dir <path>` | `<project>/results` | |
| `--system-file <path>` | `harness/prompts/system.md` | |
| `--task-file <path>` | `harness/prompts/task.md` | |

## What each case does

1. `POST /cases` on segd (`out_dir`, `nni_url`, `budgets`).
2. Builds the task message from `harness/prompts/task.md` (`{{case_id}}`,
   `{{target_text}}`, `{{sequence}}`, `{{shape}}`, `{{budgets}}`). Target text
   comes from the histotype letter (F fibrous, L lipomatous, M myxoid, N nerve
   sheath, V vascular soft-tissue tumor); sequence and split from
   `eval/v2/mstt_cases.csv`; shape from `eval/v2/data_check/design.jsonl` when
   the case is there, otherwise `unknown (seen at seg_open)`.
3. Creates isolated session services: a fresh in-memory `SettingsManager`
   (`compaction: off`, `retry: 2`), the shared `ModelRuntime`, and a
   `DefaultResourceLoader` with **all discovery disabled** (`noExtensions`,
   `noSkills`, `noPromptTemplates`, `noThemes`, `noContextFiles`) whose only
   resources are (a) the `seg_*` extension factory bound to this case's handle,
   (b) the auth shim, and (c) a diagnostic `before_provider_request` observer.
4. Activates exactly the `seg_*` tool names the extension registered (the SDK
   `tools` list does not expand globs).
5. Prompts with the task message. After every `prompt()` resolves it reads segd
   status: if not done and wall time and budget remain, it sends the nudge
   message, up to `--max-nudges` times.
6. If still not done (or on wall-time abort), it calls the `finish` op over HTTP
   with `confidence: "low"`, `reason: "forced: <why>"`, and sets
   `meta.forced_finish`.
7. Copies the session file to `session.jsonl`, `DELETE`s the case, writes
   `meta.json`.

One case failing does not stop the batch; its `meta.json` records the error and
`batch_summary.json` marks it `ok: false`.

## Output

```
results/<run_id>/
  batch_summary.json          # per-case summary + aggregate
  <case>/
    session.jsonl             # full pi session (messages, tool calls, tool results, images)
    meta.json                 # model, versions, hashes, usage, cost, turns, nudges, errors
    trajectory.jsonl          # written by segd
    renders/ versions.npz final_mask.nii.gz   # written by segd
```

`meta.json` fields of note:

- `pi_version`, `auth_shim_version`, `model`, `thinking_level`.
- `harness_sha256` — sha256 over every file under `harness/` (sorted relative
  path + bytes), excluding `node_modules/`, `__pycache__/`, `logs/`,
  `results/`, `.git/`. There is no git repo, so this is the harness version.
- `system_prompt_sha256` — the substituted system prompt we supplied.
- `system_prompt_sent_to_provider` + `..._sha256` — captured at pi's
  `before_provider_request` hook. The shim shapes the payload *after* that hook,
  so `system_prompt_shaped_by_auth_shim` re-applies the shim's own
  `shapeAnthropicOAuthPayload` to that payload and records the closest-to-wire
  text (best effort; absent if the shim internals cannot be imported).
- `tool_names_seen_by_model`, `tool_calls_by_name`, `turns`, `nudges_used`,
  `forced_finish`, `token_usage` (summed from `message_end` usage), `cost_usd`.
- `final_assistant_text`.

## Smoke test

```bash
cd harness/runner

# (a) runner plumbing + auth + inline images + tool isolation, one short session:
npx tsx run_cases.ts --cases F14 --run-id smoke-ping \
  --extension "$PWD/../checks/ping_ext.ts" --dry-stub --max-nudges 0 \
  --wall-min 5 --task-file test/smoke_ping_task.md

# (b) real extension end to end against the stub segd, with a forced nudge:
SEGD_STUB_FAIL_FIRST_FINISH=1 npx tsx run_cases.ts --cases F14 --run-id smoke-ext \
  --dry-stub --max-tool-calls 8 --max-model-prompts 4 --max-nudges 3 --wall-min 4 \
  --task-file test/smoke_ext_task.md
```

`test/stub_segd.mjs` implements `/healthz`, `POST /cases`, `POST /cases/{h}/op`
(canned text + a 24×24 PNG; `finish` sets done), `GET /cases/{h}/status`, and
`DELETE /cases/{h}`. `SEGD_STUB_FAIL_FIRST_FINISH=1` rejects each case's first
`finish`, which exercises the continuation/nudge path deterministically.

## Known assumptions / open risks

- **`seg_box` on the wire.** The tool schema takes `ids: [a, b]` (per the
  handoff), but the op is sent as `box(id_a, id_b)` to match CONTRACTS §6. If
  segd implements `box(ids)`, change the two-key args in
  `harness/extension/index.ts`.
- **`seg_paint` `slice_range`.** Sent as a two-element array `[start, stop]`;
  CONTRACTS names the argument but not its shape.
- **`--extension <path>` + concurrency.** A file path extension cannot be bound
  to a case handle (it reads `SEGD_CASE` from the environment), so only use it
  with a single case.
- **Provider payload capture** is pre-shim; the shim's rewrite is reproduced
  only best effort (see above).

## Related

- `harness/extension/index.ts` — the `seg_*` tools (`createSegExtension({segdUrl, caseHandle})`
  plus a `SEGD_URL`/`SEGD_CASE` default export).
- `harness/bin/pi-seg <case> [run_id]` — interactive wrapper that opens the case
  and launches `pi` with the same isolation and system prompt.
