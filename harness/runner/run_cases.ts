#!/usr/bin/env -S npx tsx
/**
 * pi SDK batch runner for the segmentation harness (design §8.3).
 *
 * One process, one shared ModelRuntime (so a single OAuth refresh happens at a
 * time), one AgentSession per case, a concurrency pool. Each case drives the
 * real model through the seg_* extension (or any `--extension` path for
 * smoke tests) against segd, nags the model until it calls seg_finish, and
 * records everything under `results/<run_id>/<case>/`.
 *
 * See harness/runner/README.md for how to run it.
 */
import { spawn, type ChildProcess } from "node:child_process";
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
	copyFileSync,
	existsSync,
	mkdirSync,
	readFileSync,
	readdirSync,
	renameSync,
	writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
	createAgentSessionFromServices,
	createAgentSessionServices,
	getAgentDir,
	ModelRuntime,
	SessionManager,
	SettingsManager,
	VERSION,
	type AgentSession,
	type ExtensionAPI,
	type ExtensionFactory,
} from "@earendil-works/pi-coding-agent";
import { createSegExtension, SEG_TOOL_NAMES } from "../extension/index.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(HERE, "..", "..");
const HARNESS_DIR = join(PROJECT_ROOT, "harness");
const DEFAULT_EXTENSION_DIR = join(HARNESS_DIR, "extension");
const DEFAULT_SYSTEM_FILE = join(HARNESS_DIR, "prompts", "system.md");
const DEFAULT_TASK_FILE = join(HARNESS_DIR, "prompts", "task.md");
const CASES_CSV = join(PROJECT_ROOT, "eval", "v2", "mstt_cases.csv");
const DESIGN_SHAPES = join(PROJECT_ROOT, "eval", "v2", "data_check", "design.jsonl");
const DEFAULT_AUTH_SHIM = join(
	homedir(),
	".pi",
	"agent",
	"npm",
	"node_modules",
	"@gotgenes",
	"pi-anthropic-auth",
);
const STUB_SERVER = join(HERE, "test", "stub_segd.mjs");

const HISTOTYPE_TEXT: Record<string, string> = {
	F: "fibrous soft-tissue tumor",
	L: "lipomatous soft-tissue tumor",
	M: "myxoid soft-tissue tumor",
	N: "peripheral nerve sheath tumor",
	V: "vascular soft-tissue tumor",
};

const NUDGE_MESSAGE =
	"You have not called seg_finish yet. Continue segmenting; call seg_finish when you are satisfied.";

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

interface Options {
	cases: string[];
	split: string | null;
	runId: string;
	concurrency: number;
	model: string;
	thinking: string;
	wallMin: number;
	maxNudges: number;
	maxToolCalls: number;
	maxModelPrompts: number;
	segdUrl: string;
	nniUrl: string;
	extension: string;
	dryStub: boolean;
	authShim: string;
	resultsDir: string;
	note: string;
	examples: string;
	systemFile: string;
	taskFile: string;
	resume: boolean;
}

function parseArgs(argv: string[]): Options {
	const opts: Options = {
		cases: [],
		split: null,
		runId: `run-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`,
		concurrency: 4,
		model: "anthropic/claude-sonnet-5-5",
		thinking: "medium",
		wallMin: 10,
		maxNudges: 3,
		maxToolCalls: 40,
		maxModelPrompts: 15,
		segdUrl: "http://127.0.0.1:8765",
		nniUrl: "http://127.0.0.1:1527",
		extension: DEFAULT_EXTENSION_DIR,
		dryStub: false,
		authShim: process.env.AUTH_SHIM ?? DEFAULT_AUTH_SHIM,
		resultsDir: join(PROJECT_ROOT, "results"),
		note: "",
		resume: false,
		examples: "",
		systemFile: DEFAULT_SYSTEM_FILE,
		taskFile: DEFAULT_TASK_FILE,
	};

	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		const next = (): string => {
			const value = argv[++i];
			if (value === undefined) throw new Error(`missing value for ${arg}`);
			return value;
		};
		switch (arg) {
			case "--cases":
				opts.cases = next().split(",").map((c) => c.trim()).filter(Boolean);
				break;
			case "--split":
				opts.split = next();
				break;
			case "--run-id":
				opts.runId = next();
				break;
			case "--concurrency":
				opts.concurrency = Number(next());
				break;
			case "--model":
				opts.model = next();
				break;
			case "--thinking":
				opts.thinking = next();
				break;
			case "--wall-min":
				opts.wallMin = Number(next());
				break;
			case "--max-nudges":
				opts.maxNudges = Number(next());
				break;
			case "--max-tool-calls":
				opts.maxToolCalls = Number(next());
				break;
			case "--max-model-prompts":
				opts.maxModelPrompts = Number(next());
				break;
			case "--segd-url":
				opts.segdUrl = next();
				break;
			case "--nni-url":
				opts.nniUrl = next();
				break;
			case "--extension":
				opts.extension = resolve(next());
				break;
			case "--dry-stub":
				opts.dryStub = true;
				break;
			case "--auth-shim":
				opts.authShim = resolve(next());
				break;
			case "--examples":
				opts.examples = resolve(next());
				break;
			case "--note":
				opts.note = next();
				break;
			case "--resume":
				opts.resume = true;
				break;
			case "--results-dir":
				opts.resultsDir = resolve(next());
				break;
			case "--system-file":
				opts.systemFile = resolve(next());
				break;
			case "--task-file":
				opts.taskFile = resolve(next());
				break;
			case "--help":
			case "-h":
				printHelp();
				process.exit(0);
				break;
			default:
				throw new Error(`unknown option: ${arg}`);
		}
	}
	return opts;
}

function printHelp(): void {
	console.log(`Usage: npx tsx run_cases.ts [options]

  --cases F14,M07           comma-separated case ids (or --split)
  --split design|test|all   run every case in that split of eval/v2/mstt_cases.csv
  --run-id <id>             run directory name under results/ (default: timestamp)
  --concurrency <n>         parallel cases in this process (default 4)
  --model <provider/id>     (default anthropic/claude-sonnet-5-5)
  --thinking <level>        (default medium)
  --wall-min <n>            per-case wall-time limit (default 10)
  --max-nudges <n>          follow-up "call seg_finish" messages (default 3)
  --max-tool-calls <n>      budget sent to segd (default 40)
  --max-model-prompts <n>   budget sent to segd (default 15)
  --segd-url <url>          (default http://127.0.0.1:8765)
  --nni-url <url>           (default http://127.0.0.1:1527)
  --extension <path>        default harness/extension (dir); checks/ping_ext.ts for smoke
  --dry-stub                spawn the bundled stub segd and use it as --segd-url
  --examples <manifest>     Phase 2 prepare manifest.jsonl: attach each case's leave-one-out example image
                            (another case's GT outline, 3 planes) to the task message (E12)
  --note <text>             free-text note stored in batch_summary.json (experiment log)
  --resume                  skip cases already complete in this run (meta.json done, no API errors);
                            move incomplete case dirs to <run>/_incomplete/. The runner stops pulling new
                            cases after a case hits a provider/API error and exits with code 3.
  --auth-shim <path>        auth shim package dir (default @gotgenes/pi-anthropic-auth)
  --results-dir <path>      (default <project>/results)
  --system-file <path>      (default harness/prompts/system.md)
  --task-file <path>        (default harness/prompts/task.md)`);
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function postJson(url: string, body: unknown, timeoutMs = 30_000): Promise<any> {
	return fetch(url, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(body),
		signal: AbortSignal.timeout(timeoutMs),
	}).then(async (res) => {
		const text = await res.text();
		try {
			return JSON.parse(text);
		} catch {
			throw new Error(`non-JSON reply from ${url} (HTTP ${res.status}): ${text.slice(0, 300)}`);
		}
	});
}

function getJson(url: string, timeoutMs = 15_000): Promise<any> {
	return fetch(url, { signal: AbortSignal.timeout(timeoutMs) }).then(async (res) => {
		const text = await res.text();
		try {
			return JSON.parse(text);
		} catch {
			throw new Error(`non-JSON reply from ${url} (HTTP ${res.status}): ${text.slice(0, 300)}`);
		}
	});
}

function sha256(text: string | Buffer): string {
	return createHash("sha256").update(text).digest("hex");
}

function renderTemplate(text: string, vars: Record<string, string>): string {
	return text.replace(/\{\{(\w+)\}\}/g, (match, key: string) =>
		Object.prototype.hasOwnProperty.call(vars, key) ? vars[key] : match,
	);
}

function walkFiles(root: string, skipSegments: Set<string>): string[] {
	const out: string[] = [];
	const visit = (dir: string): void => {
		for (const entry of readdirSync(dir, { withFileTypes: true })) {
			if (skipSegments.has(entry.name)) continue;
			const full = join(dir, entry.name);
			if (entry.isDirectory()) visit(full);
			else if (entry.isFile()) out.push(full);
		}
	};
	visit(root);
	return out;
}

/** git provenance of the code that runs (commit, tag/describe, dirty files under harness/ and eval/v2/). */
function gitInfo(): { commit: string | null; describe: string | null; dirty: string[] } {
	const run = (cmd: string): string | null => {
		try {
			return execSync(cmd, { cwd: PROJECT_ROOT, stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
		} catch {
			return null;
		}
	};
	const status = run("git status --porcelain -- harness eval/v2");
	return {
		commit: run("git rev-parse HEAD"),
		describe: run("git describe --tags --always --dirty"),
		dirty: status ? status.split("\n").filter((l) => l.trim()) : [],
	};
}

/** sha256 over the harness source tree (excludes venvs/caches), kept alongside the git commit. */
function hashHarnessTree(): string {
	const skip = new Set(["node_modules", "__pycache__", "logs", "results", ".git", ".venv", ".pytest_cache"]);
	const files = walkFiles(HARNESS_DIR, skip).sort();
	const hash = createHash("sha256");
	for (const file of files) {
		hash.update(relative(HARNESS_DIR, file));
		hash.update("\0");
		hash.update(readFileSync(file));
		hash.update("\0");
	}
	return hash.digest("hex");
}

function readJsonIfExists(path: string): any | undefined {
	try {
		if (!existsSync(path)) return undefined;
		return JSON.parse(readFileSync(path, "utf8"));
	} catch {
		return undefined;
	}
}

interface CaseRow {
	case: string;
	histotype: string;
	sequence: string;
	split: string;
}

function loadCases(): CaseRow[] {
	const raw = readFileSync(CASES_CSV, "utf8").trim().split(/\r?\n/);
	const header = raw[0].split(",");
	const rows = raw.slice(1).map((line) => {
		const cells = line.split(",");
		const row: Record<string, string> = {};
		header.forEach((name, index) => {
			row[name] = cells[index];
		});
		return row as unknown as CaseRow;
	});
	return rows;
}

function shapeForCase(caseId: string): string {
	let shapes: any[] | undefined;
	try {
		shapes = [DESIGN_SHAPES, DESIGN_SHAPES.replace(/design\.jsonl$/, "test.jsonl")]
			.filter((p) => existsSync(p))
			.flatMap((p) => readFileSync(p, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l)));
	} catch {
		shapes = undefined;
	}
	if (Array.isArray(shapes)) {
		const row = shapes.find((r) => r?.case === caseId);
		if (row?.shape) return (row.shape as number[]).join("x");
	}
	return "unknown (seen at seg_open)";
}

function authShimVersion(shimPath: string): string | undefined {
	const pkg = readJsonIfExists(join(shimPath, "package.json"));
	return pkg?.version;
}

function extractSystemText(payload: unknown): string | undefined {
	if (!payload || typeof payload !== "object") return undefined;
	const system = (payload as { system?: unknown }).system;
	if (typeof system === "string") return system;
	if (Array.isArray(system)) {
		const parts = system
			.map((block) => (block && typeof block === "object" ? (block as any).text : undefined))
			.filter((text): text is string => typeof text === "string");
		if (parts.length > 0) return parts.join("\n\n");
	}
	return undefined;
}

/**
 * Best-effort application of the auth shim's own OAuth payload shaping to a
 * captured provider payload, so meta.json can record the system prompt closest
 * to the wire. The shim shapes inside its transport (after pi's
 * `before_provider_request` hook), so this reproduces that step. Returns
 * undefined when the shim's internals cannot be imported.
 */
async function shapeWithShim(
	shimPath: string,
	payload: unknown,
): Promise<unknown | undefined> {
	try {
		const modulePath = join(shimPath, "src", "request-shaping.ts");
		if (!existsSync(modulePath)) return undefined;
		const mod = (await import(pathToFileURL(modulePath).href)) as {
			shapeAnthropicOAuthPayload?: (p: unknown) => unknown;
		};
		return mod.shapeAnthropicOAuthPayload?.(payload);
	} catch {
		return undefined;
	}
}

// ---------------------------------------------------------------------------
// Stub segd
// ---------------------------------------------------------------------------

async function startStubServer(): Promise<{ proc: ChildProcess; url: string }> {
	const proc = spawn(process.execPath, [STUB_SERVER, "0"], {
		env: process.env,
		stdio: ["ignore", "pipe", "pipe"],
	});
	const stdout = proc.stdout;
	if (!stdout) throw new Error("stub segd has no stdout");
	const url = await new Promise<string>((resolvePromise, reject) => {
		let buffer = "";
		const timer = setTimeout(() => reject(new Error("stub segd did not start")), 10_000);
		stdout.on("data", (chunk: Buffer) => {
			buffer += chunk.toString();
			const line = buffer.split("\n").find((l) => l.includes('"ready"'));
			if (line) {
				clearTimeout(timer);
				try {
					resolvePromise(JSON.parse(line).url as string);
				} catch (error) {
					reject(error as Error);
				}
			}
		});
		proc.on("error", (error) => {
			clearTimeout(timer);
			reject(error);
		});
	});
	return { proc, url };
}

// ---------------------------------------------------------------------------
// Per-case run
// ---------------------------------------------------------------------------

interface CaseResult {
	case: string;
	ok: boolean;
	done: boolean;
	forced_finish: boolean;
	meta_path: string;
	wall_time_s: number;
	turns: number;
	nudges: number;
	errors: string[];
	api_errors?: string[];
}

interface Shared {
	modelRuntime: ModelRuntime;
	harnessHash: string;
	git: { commit: string | null; describe: string | null; dirty: string[] };
	systemPrompt: string;
	systemPromptSha: string;
	shimVersion?: string;
}

async function runCase(row: CaseRow, opts: Options, shared: Shared): Promise<CaseResult> {
	const caseId = row.case;
	const caseDir = join(opts.resultsDir, opts.runId, caseId);
	mkdirSync(caseDir, { recursive: true });

	const errors: string[] = [];
	const startedAt = Date.now();
	let session: AgentSession | undefined;
	const capture: { seen: boolean; payload?: unknown; system?: string } = { seen: false };

	const isDefaultExtension = resolve(opts.extension) === resolve(DEFAULT_EXTENSION_DIR);
	let sessionFileCopied = false;

	const result: CaseResult = {
		case: caseId,
		ok: false,
		done: false,
		forced_finish: false,
		meta_path: join(caseDir, "meta.json"),
		wall_time_s: 0,
		turns: 0,
		nudges: 0,
		errors,
	};

	try {
		// 1. Open the case on segd.
		const opened = await postJson(`${opts.segdUrl.replace(/\/+$/, "")}/cases`, {
			case_id: caseId,
			run_id: opts.runId,
			// segd writes to <out_dir>/<run_id>/<case_id> (= caseDir) and caches marks
			// at <out_dir>/cache/marks, so send the results root here.
			out_dir: opts.resultsDir,
			nni_url: opts.nniUrl,
			budgets: {
				max_tool_calls: opts.maxToolCalls,
				max_model_prompts: opts.maxModelPrompts,
			},
		});
		if (opened?.ok === false) throw new Error(`POST /cases failed: ${opened.error ?? "unknown"}`);

		const handle = opened?.handle ?? caseId;

		// 2. Task message for this case.
		const taskTemplate = readFileSync(opts.taskFile, "utf8");
		const taskMessage = renderTemplate(taskTemplate, {
			case_id: caseId,
			target_text: HISTOTYPE_TEXT[row.histotype] ?? row.histotype,
			sequence: row.sequence,
			shape: shapeForCase(caseId),
			budgets: `max ${opts.maxToolCalls} tool calls / ${opts.maxModelPrompts} model prompts / ${opts.wallMin} min wall time`,
		});

		// 2b. Optional example image (E12): another design case of the same histotype with its GT outline.
		let exampleImages: { type: "image"; data: string; mimeType: string }[] = [];
		let exampleText = "";
		if (opts.examples) {
			const rows = readFileSync(opts.examples, "utf8").split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
			const ex = rows.find((r: any) => r.case === caseId);
			if (!ex?.image_example) throw new Error(`no example image for ${caseId} in ${opts.examples}`);
			const exPath = resolve(PROJECT_ROOT, ex.image_example);
			const buf = readFileSync(exPath);
			exampleImages = [{ type: "image", data: buf.toString("base64"), mimeType: "image/png" }];
			exampleText =
				`\n\nThe attached image is an example from a different patient (not this case): a ` +
				`${HISTOTYPE_TEXT[row.histotype] ?? row.histotype} outlined in green in axial, coronal and sagittal slices. ` +
				"Use it to recognize this tumor type and to judge how much tissue the outline should include.";
			mkdirSync(caseDir, { recursive: true });
			writeFileSync(
				join(caseDir, "example.json"),
				JSON.stringify({ manifest: opts.examples, image: ex.image_example, sha256: createHash("sha256").update(buf).digest("hex"),
					donor: ex.example_donor ?? null, note: ex.example_note ?? null, text: exampleText.trim() }, null, 2) + "\n",
			);
		}

		// 3. Extension set: our seg_* factory bound to this case (or an override
		//    path), plus the auth shim, plus a payload-capture factory.
		const captureFactory: ExtensionFactory = (pi: ExtensionAPI) => {
			pi.on("before_provider_request", (event) => {
				capture.seen = true;
				capture.payload = event.payload;
				capture.system = extractSystemText(event.payload) ?? capture.system;
			});
		};

		const extensionFactories: ExtensionFactory[] = [captureFactory];
		const additionalExtensionPaths: string[] = [resolve(opts.authShim)];
		if (isDefaultExtension) {
			extensionFactories.unshift(
				createSegExtension({ segdUrl: opts.segdUrl, caseHandle: handle }),
			);
		} else {
			additionalExtensionPaths.push(resolve(opts.extension));
			process.env.SEGD_CASE = handle;
			process.env.SEGD_URL = opts.segdUrl;
		}

		// 4. Fresh settings per case; shared ModelRuntime.
		const settingsManager = SettingsManager.inMemory({
			compaction: { enabled: false },
			retry: { enabled: true, maxRetries: 2 },
		});

		const services = await createAgentSessionServices({
			cwd: caseDir,
			agentDir: getAgentDir(),
			modelRuntime: shared.modelRuntime,
			settingsManager,
			resourceLoaderOptions: {
				noExtensions: true,
				noSkills: true,
				noPromptTemplates: true,
				noThemes: true,
				noContextFiles: true,
				additionalExtensionPaths,
				extensionFactories,
				systemPrompt: shared.systemPrompt,
			},
		});
		for (const diagnostic of services.diagnostics) {
			if (diagnostic.type === "error") errors.push(`services: ${diagnostic.message}`);
		}
		for (const loadError of services.resourceLoader.getExtensions().errors) {
			errors.push(`extension load: ${loadError.path}: ${loadError.error}`);
		}

		// 5. Model + thinking.
		const [provider, ...modelParts] = opts.model.split("/");
		const modelId = modelParts.join("/");
		const model = shared.modelRuntime.getModel(provider, modelId);
		if (!model) throw new Error(`model not found: ${opts.model}`);

		// 6. Persistent session file inside the case directory.
		const sessionManager = SessionManager.create(caseDir, caseDir);

		// 6b. Activate exactly the seg_* tools the loaded extensions registered.
		//     (The SDK `tools` list does not expand globs, so resolve names here.)
		const registeredSegTools = [
			...new Set(
				services.resourceLoader
					.getExtensions()
					.extensions.flatMap((extension) => [...extension.tools.keys()])
					.filter((name) => name.startsWith("seg_")),
			),
		].sort();
		const tools = registeredSegTools.length > 0 ? registeredSegTools : [...SEG_TOOL_NAMES];
		if (registeredSegTools.length === 0) {
			errors.push("no seg_* tools were registered by the loaded extensions");
		}
		const created = await createAgentSessionFromServices({
			services,
			sessionManager,
			model,
			thinkingLevel: opts.thinking as any,
			tools,
		});
		session = created.session;
		const declaredToolNames = session.getActiveToolNames();
		console.log(`[${caseId}] tools declared to model: ${declaredToolNames.join(", ") || "(none)"}`);
		result.turns = 0;

		// 7. Usage / turn accounting.
		let inputTokens = 0;
		let outputTokens = 0;
		let cacheRead = 0;
		let cacheWrite = 0;
		let costUsd = 0;
		const toolCallsByName: Record<string, number> = {};
		const apiErrors: string[] = [];
		session.subscribe((event) => {
			if (event.type === "message_end" && event.message.role === "assistant") {
				result.turns += 1;
				const m = event.message as any;
				if (m.stopReason === "error") apiErrors.push(String(m.errorMessage ?? "assistant stopReason=error"));
				const usage = event.message.usage;
				if (usage) {
					inputTokens += usage.input ?? 0;
					outputTokens += usage.output ?? 0;
					cacheRead += usage.cacheRead ?? 0;
					cacheWrite += usage.cacheWrite ?? 0;
					costUsd += usage.cost?.total ?? 0;
				}
			}
			if (event.type === "tool_execution_start") {
				toolCallsByName[event.toolName] = (toolCallsByName[event.toolName] ?? 0) + 1;
			}
		});

		// 8. Per-case wall clock.
		let aborted = false;
		const wallTimer = setTimeout(() => {
			aborted = true;
			void session?.abort().catch(() => {});
		}, opts.wallMin * 60_000);

		const statusOf = async (): Promise<any | undefined> => {
			try {
				return await getJson(`${opts.segdUrl.replace(/\/+$/, "")}/cases/${encodeURIComponent(handle)}/status`);
			} catch (error) {
				errors.push(`status: ${(error as Error).message}`);
				return undefined;
			}
		};

		// 9. Prompt loop: keep going until seg_finish (done) or a stop condition.
		let forcedReason: string | undefined;
		try {
			if (exampleImages.length) await session.prompt(taskMessage + exampleText, { images: exampleImages });
			else await session.prompt(taskMessage);
			for (;;) {
				const status = await statusOf();
				if (status?.done) break;
				if (aborted) {
					forcedReason = "wall time exceeded";
					break;
				}
				const withinBudget =
					(status?.tool_calls ?? 0) < opts.maxToolCalls &&
					(status?.model_prompts ?? 0) < opts.maxModelPrompts;
				if (!withinBudget) {
					forcedReason = "budget exhausted";
					break;
				}
				if (result.nudges >= opts.maxNudges) {
					forcedReason = "max nudges reached";
					break;
				}
				result.nudges += 1;
				await session.prompt(NUDGE_MESSAGE);
			}
		} catch (error) {
			errors.push(`prompt: ${(error as Error).message}`);
			if (!forcedReason) forcedReason = "prompt error";
		}
		clearTimeout(wallTimer);

		// 10. Force-finish when the model never finished.
		let finalStatus = await statusOf();
		if (!finalStatus?.done) {
			result.forced_finish = true;
			const reason = forcedReason ?? "not done after max nudges";
			try {
				await postJson(
					`${opts.segdUrl.replace(/\/+$/, "")}/cases/${encodeURIComponent(handle)}/op`,
					{ op: "finish", args: { confidence: "low", reason: `forced: ${reason}`, skip_review: true } },
				);
			} catch (error) {
				errors.push(`forced finish: ${(error as Error).message}`);
			}
			finalStatus = await statusOf();
		}
		result.done = finalStatus?.done === true;
		const finalAssistantText = session.getLastAssistantText();

		// 11. Copy the session file to a stable name, then release the case.
		const rawSessionFile = sessionManager.getSessionFile();
		session.dispose();
		session = undefined;
		if (rawSessionFile && existsSync(rawSessionFile)) {
			const dest = join(caseDir, "session.jsonl");
			if (resolve(rawSessionFile) !== resolve(dest)) {
				try {
					renameSync(rawSessionFile, dest);
				} catch {
					copyFileSync(rawSessionFile, dest);
				}
			}
			sessionFileCopied = existsSync(dest);
		}

		try {
			await fetch(
				`${opts.segdUrl.replace(/\/+$/, "")}/cases/${encodeURIComponent(handle)}`,
				{ method: "DELETE", signal: AbortSignal.timeout(15_000) },
			);
		} catch (error) {
			errors.push(`DELETE case: ${(error as Error).message}`);
		}

		// 12. Record the system prompt that reached the provider (best effort).
		let providerSystemPrompt: string | undefined = capture.system;
		let providerSystemPromptShaped: string | undefined;
		if (capture.payload !== undefined) {
			const shaped = await shapeWithShim(opts.authShim, capture.payload);
			providerSystemPromptShaped = extractSystemText(shaped);
		}

		const wallTimeS = (Date.now() - startedAt) / 1000;
		result.wall_time_s = wallTimeS;
		// provider errors only invalidate the case when the agent did not finish on its own
		// (pi auto-retries transient errors; a recovered case is a valid result)
		// (the wall-time abort surfaces as "This operation was aborted": that is a budget outcome, not a provider error)
		const fatalApiErrors = result.forced_finish && !aborted ? apiErrors : [];
		if (fatalApiErrors.length) errors.push(...fatalApiErrors.map((e) => `api: ${e}`));
		result.api_errors = fatalApiErrors;
		result.ok = errors.length === 0;

		const meta = {
			case_id: caseId,
			run_id: opts.runId,
			model: opts.model,
			thinking_level: opts.thinking,
			pi_version: VERSION,
			auth_shim_path: resolve(opts.authShim),
			auth_shim_version: shared.shimVersion,
			harness_sha256: shared.harnessHash,
			git_commit: shared.git.commit,
			git_describe: shared.git.describe,
			git_dirty_files: shared.git.dirty,
			harness_hash_method:
				"sha256 over sorted relative paths + bytes of every file under harness/ excluding node_modules/, __pycache__/, logs/, results/, .git/",
			system_prompt_path: opts.systemFile,
			system_prompt_sha256: shared.systemPromptSha,
			system_prompt_sent_to_provider_captured: capture.seen,
			system_prompt_sent_to_provider: providerSystemPrompt,
			system_prompt_sent_to_provider_sha256: providerSystemPrompt
				? sha256(providerSystemPrompt)
				: undefined,
			system_prompt_shaped_by_auth_shim: providerSystemPromptShaped,
			system_prompt_shaped_by_auth_shim_sha256: providerSystemPromptShaped
				? sha256(providerSystemPromptShaped)
				: undefined,
			system_prompt_notes:
				"captured at pi's before_provider_request hook (pre-shim); the shaped field applies @gotgenes/pi-anthropic-auth request shaping, which the shim performs after that hook",
			task_message: taskMessage,
			tools_declared: tools,
			tool_names_seen_by_model: declaredToolNames,
			tool_calls_by_name: toolCallsByName,
			session_file: sessionFileCopied ? "session.jsonl" : undefined,
			wall_time_s: wallTimeS,
			turns: result.turns,
			nudges_used: result.nudges,
			forced_finish: result.forced_finish,
			done: result.done,
			final_assistant_text: finalAssistantText,
			token_usage: {
				input: inputTokens,
				output: outputTokens,
				cache_read: cacheRead,
				cache_write: cacheWrite,
				total: inputTokens + outputTokens,
			},
			cost_usd: costUsd,
			errors,
			api_errors: fatalApiErrors,
			api_errors_all: apiErrors,
			started_at: new Date(startedAt).toISOString(),
			finished_at: new Date().toISOString(),
		};
		writeFileSync(join(caseDir, "meta.json"), `${JSON.stringify(meta, null, 2)}\n`);

		console.log(
			`[${caseId}] done=${result.done} forced=${result.forced_finish} turns=${result.turns} ` +
				`nudges=${result.nudges} tools=${JSON.stringify(toolCallsByName)} wall=${wallTimeS.toFixed(1)}s`,
		);
		return result;
	} catch (error) {
		errors.push((error as Error).message);
		result.ok = false;
		result.wall_time_s = (Date.now() - startedAt) / 1000;
		try {
			session?.dispose();
		} catch {
			// ignore
		}
		writeFileSync(
			join(caseDir, "meta.json"),
			`${JSON.stringify({ case_id: caseId, run_id: opts.runId, errors, ok: false }, null, 2)}\n`,
		);
		console.error(`[${caseId}] failed: ${(error as Error).message}`);
		return result;
	}
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
	const opts = parseArgs(process.argv.slice(2));

	let stub: { proc: ChildProcess; url: string } | undefined;
	if (opts.dryStub) {
		stub = await startStubServer();
		opts.segdUrl = stub.url;
		console.log(`[runner] dry-stub segd at ${stub.url}`);
	}

	const allRows = loadCases();
	let rows: CaseRow[];
	if (opts.cases.length > 0) {
		rows = allRows.filter((row) => opts.cases.includes(row.case));
		const missing = opts.cases.filter((c) => !allRows.some((r) => r.case === c));
		if (missing.length > 0) throw new Error(`unknown cases: ${missing.join(", ")}`);
	} else if (opts.split) {
		rows = opts.split === "all" ? allRows : allRows.filter((row) => row.split === opts.split);
	} else {
		throw new Error("pass --cases or --split");
	}
	if (rows.length === 0) throw new Error("no cases selected");

	const systemTemplate = readFileSync(opts.systemFile, "utf8");
	const systemPrompt = renderTemplate(systemTemplate, {
		max_tool_calls: String(opts.maxToolCalls),
		max_model_prompts: String(opts.maxModelPrompts),
	});

	const modelRuntime = await ModelRuntime.create({
		authPath: join(getAgentDir(), "auth.json"),
	});
	const [provider, ...modelParts] = opts.model.split("/");
	const modelId = modelParts.join("/");
	if (!modelRuntime.getModel(provider, modelId)) {
		throw new Error(`model not found: ${opts.model}`);
	}

	const shared: Shared = {
		modelRuntime,
		harnessHash: hashHarnessTree(),
		git: gitInfo(),
		systemPrompt,
		systemPromptSha: sha256(systemPrompt),
		shimVersion: authShimVersion(opts.authShim),
	};

	console.log(
		`[runner] run=${opts.runId} cases=${rows.map((r) => r.case).join(",")} concurrency=${opts.concurrency} ` +
			`model=${opts.model} thinking=${opts.thinking} extension=${opts.extension} segd=${opts.segdUrl}`,
	);
	console.log(`[runner] harness_sha256=${shared.harnessHash} git=${shared.git.describe} dirty_files=${shared.git.dirty.length}`);
	console.log(`[runner] system_prompt_sha256=${shared.systemPromptSha} auth_shim=${shared.shimVersion}`);

	if (opts.resume) {
		const runDir = join(opts.resultsDir, opts.runId);
		const todo: CaseRow[] = [];
		for (const row of rows) {
			const dir = join(runDir, row.case);
			const meta = readJsonIfExists(join(dir, "meta.json"));
			const complete =
				meta?.done === true &&
				(meta.api_errors ?? []).length === 0 &&
				(meta.errors ?? []).length === 0 &&
				existsSync(join(dir, "final_mask.nii.gz"));
			if (complete) continue;
			// count only genuine agent-run failures (the agent ran to the end but the case recorded errors);
			// interrupted/killed attempts and infrastructure failures (no `done` in meta) never count
			const isAgentFailure = (m: any) =>
				m?.done !== undefined && m?.turns !== undefined && (m.api_errors ?? []).length === 0 && (m.errors ?? []).length > 0;
			const parked = existsSync(join(runDir, "_incomplete"))
				? readdirSync(join(runDir, "_incomplete"))
						.filter((n) => n.startsWith(`${row.case}_`) && !n.startsWith(`${row.case}_api_`))
						.filter((n) => isAgentFailure(readJsonIfExists(join(runDir, "_incomplete", n, "meta.json")))).length
				: 0;
			if (parked >= 3 && isAgentFailure(meta)) {
				// 4 non-provider attempts failed (3 parked + current): keep the last attempt as the result, stop retrying
				console.log(`[runner] resume: giving up on ${row.case} after ${parked + 1} attempts (kept last attempt)`);
				continue;
			}
			if (existsSync(dir)) {
				const apiFail = (meta?.api_errors ?? []).length > 0;
				const dest = join(runDir, "_incomplete", `${row.case}_${apiFail ? "api_" : ""}${Date.now()}`);
				mkdirSync(join(runDir, "_incomplete"), { recursive: true });
				renameSync(dir, dest);
				console.log(`[runner] resume: moved incomplete ${row.case} -> ${dest}`);
			}
			todo.push(row);
		}
		console.log(`[runner] resume: ${rows.length - todo.length} complete, ${todo.length} to run`);
		rows = todo;
		if (rows.length === 0) {
			console.log("[runner] resume: nothing to do");
			return;
		}
	}

	const queue = [...rows];
	const results: CaseResult[] = [];
	const batchStarted = Date.now();
	let stopReason: string | undefined;

	const worker = async (): Promise<void> => {
		for (;;) {
			if (stopReason) return;
			const row = queue.shift();
			if (!row) return;
			const r = await runCase(row, opts, shared);
			results.push(r);
			if (r.api_errors && r.api_errors.length) {
				stopReason = `${row.case}: ${r.api_errors[0]}`;
				console.log(`[runner] provider/API error -> stop pulling new cases (${stopReason})`);
			}
		}
	};
	const workerCount = Math.max(1, Math.min(opts.concurrency, queue.length));
	await Promise.all(Array.from({ length: workerCount }, () => worker()));

	const summary = {
		run_id: opts.runId,
		model: opts.model,
		thinking_level: opts.thinking,
		extension: opts.extension,
		segd_url: opts.segdUrl,
		nni_url: opts.nniUrl,
		concurrency: opts.concurrency,
		harness_sha256: shared.harnessHash,
		git_commit: shared.git.commit,
		git_describe: shared.git.describe,
		git_dirty_files: shared.git.dirty,
		note: opts.note,
		examples: opts.examples || null,
		system_prompt_sha256: shared.systemPromptSha,
		auth_shim_version: shared.shimVersion,
		pi_version: VERSION,
		cases: results,
		started_at: new Date(batchStarted).toISOString(),
		finished_at: new Date().toISOString(),
		wall_time_s: (Date.now() - batchStarted) / 1000,
	};
	const summaryDir = join(opts.resultsDir, opts.runId);
	mkdirSync(summaryDir, { recursive: true });
	(summary as any).stopped_on_api_error = stopReason ?? null;
	const summaryPath = join(summaryDir, "batch_summary.json");
	if (existsSync(summaryPath)) {
		// resumed run: keep earlier batch summaries, carry the earliest start time forward
		const prev = readJsonIfExists(summaryPath);
		renameSync(summaryPath, join(summaryDir, `batch_summary.${Date.now()}.json`));
		if (prev?.started_at) (summary as any).first_started_at = prev.first_started_at ?? prev.started_at;
	}
	writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
	console.log(`[runner] wrote ${summaryPath}`);

	if (stub) stub.proc.kill();
	if (stopReason) process.exit(3);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});
