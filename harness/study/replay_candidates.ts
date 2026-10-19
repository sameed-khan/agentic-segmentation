#!/usr/bin/env -S npx tsx
/**
 * E16 candidate-decision replay study: replay the first-stage seg_finish review
 * (review grid + candidate sheet) that an agent saw in a phase1 run and ask the
 * model — in a single tool-less turn — to accept/reject each candidate proposal
 * (E16; build script harness/study/prepare_candidate_replay.py).
 *
 * Same pi SDK plumbing as harness/study/ask.ts — one shared ModelRuntime
 * (subscription auth from ~/.pi/agent/auth.json, REQUIRED auth shim
 * @gotgenes/pi-anthropic-auth), fresh AgentSession per call, isolated resources
 * (no skills/context files/prompt templates/discovered extensions), compaction
 * off, ZERO tools, no agent loop: one user message (prompt text + the two review
 * PNGs), one reply, recorded raw. Strict answer parsing happens in Python
 * (harness/study/score_candidate_replay.py), never here.
 *
 * Input manifest: results/candidate_replay/manifest.jsonl. GT-derived fields
 * (precision, label, *_vox, dice_reviewed) are NEVER sent to the model — the user
 * message carries only case, target, sequence and the GT-free review text.
 *
 * Conditions: base -> prompts/agent_system_v2rc1.md, lesion ->
 * prompts/agent_system_v2rc1_lesionline.md, both with {{max_tool_calls}} -> 40
 * and {{max_model_prompts}} -> 15 substituted.
 *
 * Output: results/candidate_replay/<run>/responses.jsonl (one line per
 * key×condition×rep, resumable: completed keys are skipped), meta.json,
 * sessions/<key>_<cond>_r<rep>.jsonl.
 *
 * Run:  npx tsx replay_candidates.ts --run e16 --conditions base,lesion --reps 3
 *                       [--keys F14_v1,...] [--concurrency 4]
 */
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
	appendFileSync,
	existsSync,
	mkdirSync,
	readFileSync,
	renameSync,
	writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { ImageContent } from "@earendil-works/pi-ai";
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
	type ModelRuntime as ModelRuntimeType,
} from "@earendil-works/pi-coding-agent";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(HERE, "..", "..");
const MANIFEST_PATH = join(PROJECT_ROOT, "results", "candidate_replay", "manifest.jsonl");
const PROMPTS_DIR = join(HERE, "prompts");
const DEFAULT_AUTH_SHIM = join(
	homedir(),
	".pi",
	"agent",
	"npm",
	"node_modules",
	"@gotgenes",
	"pi-anthropic-auth",
);

// Exact target strings — keep in sync with harness/runner/run_cases.ts
// HISTOTYPE_TEXT (the runner's task messages use these).
const HISTOTYPE_TEXT: Record<string, string> = {
	F: "fibrous soft-tissue tumor",
	L: "lipomatous soft-tissue tumor",
	M: "myxoid soft-tissue tumor",
	N: "peripheral nerve sheath tumor",
	V: "vascular soft-tissue tumor",
};

// condition -> system prompt template; user template is replay_user.md for both.
const CONDITION_SPEC: Record<string, { systemTemplate: string }> = {
	base: { systemTemplate: "agent_system_v2rc1.md" },
	lesion: { systemTemplate: "agent_system_v2rc1_lesionline.md" },
};
const ALL_CONDITIONS = ["base", "lesion"] as const;
const USER_TEMPLATE = "replay_user.md";
// The replayed session ran with these budgets; the system prompt states them.
const SYSTEM_VARS: Record<string, string> = { max_tool_calls: "40", max_model_prompts: "15" };

// ---------------------------------------------------------------------------
// Types / options
// ---------------------------------------------------------------------------
interface ManifestCandidate {
	id: string;
	volume_ml: number;
	precision: number; // GT-derived: never sent to the model
	label: string; // GT-derived: never sent to the model
	region_vox: number; // GT-free, but never sent either
	inter_vox: number; // GT-derived: never sent to the model
	overlap_mask_vox: number;
}

interface ManifestRow {
	key: string;
	case: string;
	histotype: string;
	sequence: string;
	version: string;
	dice_reviewed: number; // GT-derived: never sent to the model
	mask_vox: number;
	gt_vox: number; // GT-derived: never sent to the model
	mask_gt_vox: number; // GT-derived: never sent to the model
	images: string[]; // [review grid, candidate sheet], relative to project root
	review_text: string;
	candidates: ManifestCandidate[];
	agent_accepted: string[];
	source_run: string;
}

interface Options {
	conditions: string[];
	reps: number;
	run: string;
	keys: string[];
	concurrency: number;
	model: string;
	thinking: string;
	wallMin: number;
	authShim: string;
}

function parseArgs(argv: string[]): Options {
	const opts: Options = {
		conditions: ["base", "lesion"],
		reps: 3,
		run: "",
		keys: [],
		concurrency: 4,
		model: "anthropic/claude-sonnet-5-5",
		thinking: "high",
		wallMin: 5,
		authShim: process.env.AUTH_SHIM ?? DEFAULT_AUTH_SHIM,
	};
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		const next = (): string => {
			const value = argv[++i];
			if (value === undefined) throw new Error(`missing value for ${arg}`);
			return value;
		};
		switch (arg) {
			case "--conditions":
				opts.conditions = next().split(",").map((c) => c.trim()).filter(Boolean);
				break;
			case "--reps":
				opts.reps = Number(next());
				break;
			case "--run":
				opts.run = next();
				break;
			case "--keys":
				opts.keys = next().split(",").map((c) => c.trim()).filter(Boolean);
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
			case "--auth-shim":
				opts.authShim = resolve(next());
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
	if (!opts.run) throw new Error("--run <name> is required");
	if (opts.reps < 1) throw new Error("--reps must be >= 1");
	for (const cond of opts.conditions) {
		if (!CONDITION_SPEC[cond]) throw new Error(`unknown condition: ${cond} (use ${ALL_CONDITIONS.join(", ")})`);
	}
	return opts;
}

function printHelp(): void {
	console.log(`Usage: npx tsx replay_candidates.ts --run <name> [options]

  --run <name>              run directory under results/candidate_replay/ (required)
  --conditions base,lesion  comma list from ${ALL_CONDITIONS.join(", ")} (default base,lesion)
  --reps <n>                repeats per key x condition (default 3)
  --keys F14_v1,...         subset of manifest keys (default: all in the manifest)
  --concurrency <n>         calls in flight in this process (default 4)
  --model <provider/id>     (default anthropic/claude-sonnet-5-5)
  --thinking <level>        (default high)
  --wall-min <n>            per-call wall-time abort (default 5)
  --auth-shim <path>        default ~/.pi/agent/npm/node_modules/@gotgenes/pi-anthropic-auth`);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function sha256File(path: string): string {
	return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function sha256Text(text: string): string {
	return createHash("sha256").update(text).digest("hex");
}

/** {{var}} substitution, same convention as harness/runner/run_cases.ts. */
function renderTemplate(text: string, vars: Record<string, string>): string {
	return text.replace(/\{\{(\w+)\}\}/g, (match, key: string) =>
		Object.prototype.hasOwnProperty.call(vars, key) ? vars[key] : match,
	);
}

function rel(path: string): string {
	return relative(PROJECT_ROOT, resolve(path));
}

function imageContent(path: string): ImageContent {
	return { type: "image", data: readFileSync(resolve(PROJECT_ROOT, path)).toString("base64"), mimeType: "image/png" };
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

/** Best-effort reproduction of the shim's OAuth payload shaping (see run_cases.ts). */
async function shapeWithShim(shimPath: string, payload: unknown): Promise<unknown | undefined> {
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

interface AssistantCapture {
	text?: string;
	thinking?: string;
	usage?: any;
	stopReason?: string;
	errorMessage?: string;
	model?: string;
	responseModel?: string;
	durationMs?: number;
}

// ---------------------------------------------------------------------------
// One call
// ---------------------------------------------------------------------------
interface CallKey {
	key: string;
	condition: string;
	rep: number;
}

async function runCall(
	callKey: CallKey,
	row: ManifestRow,
	opts: Options,
	shared: Shared,
): Promise<void> {
	const { key, condition, rep } = callKey;
	const spec = CONDITION_SPEC[condition];
	const target = HISTOTYPE_TEXT[row.histotype] ?? row.histotype;
	const vars: Record<string, string> = {
		case: row.case,
		target,
		sequence: row.sequence,
		review_text: row.review_text,
	};
	const userTemplate = readFileSync(join(PROMPTS_DIR, USER_TEMPLATE), "utf8");
	const promptText = renderTemplate(userTemplate, vars);
	const images = row.images.map(imageContent);

	const runDir = join(PROJECT_ROOT, "results", "candidate_replay", opts.run);
	const callDir = join(runDir, "sessions");
	mkdirSync(callDir, { recursive: true });
	const sessionName = `${key}_${condition}_r${rep}`;

	const capture: { seen: boolean; payload?: unknown; system?: string } = { seen: false };
	const captureFactory: ExtensionFactory = (pi: ExtensionAPI) => {
		pi.on("before_provider_request", (event) => {
			capture.seen = true;
			capture.payload = event.payload;
			capture.system = extractSystemText(event.payload) ?? capture.system;
		});
	};

	const errors: string[] = [];
	const retries: { attempt: number; errorMessage: string }[] = [];
	const usageSum = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, costUsd: 0, reasoning: 0 };
	const msg: AssistantCapture = {};
	let session: AgentSession | undefined;

	try {
		const settingsManager = SettingsManager.inMemory({
			compaction: { enabled: false },
			retry: { enabled: true, maxRetries: 2 },
		});
		const services = await createAgentSessionServices({
			cwd: runDir,
			agentDir: getAgentDir(),
			modelRuntime: shared.modelRuntime,
			settingsManager,
			resourceLoaderOptions: {
				noExtensions: true,
				noSkills: true,
				noPromptTemplates: true,
				noThemes: true,
				noContextFiles: true,
				additionalExtensionPaths: [resolve(opts.authShim)],
				extensionFactories: [captureFactory],
				systemPrompt: shared.systemPrompts[condition],
			},
		});
		for (const diagnostic of services.diagnostics) {
			if (diagnostic.type === "error") errors.push(`services: ${diagnostic.message}`);
		}

		const [provider, ...modelParts] = opts.model.split("/");
		const model = shared.modelRuntime.getModel(provider, modelParts.join("/"));
		if (!model) throw new Error(`model not found: ${opts.model}`);

		const sessionManager = SessionManager.create(callDir, callDir);
		const created = await createAgentSessionFromServices({
			services,
			sessionManager,
			model,
			thinkingLevel: opts.thinking as any,
			tools: [], // zero tools: the replay cannot act on its decisions
		});
		session = created.session;
		const activeTools = session.getActiveToolNames();
		if (activeTools.length > 0) errors.push(`tools active despite tools:[]: ${activeTools.join(", ")}`);

		session.subscribe((event: any) => {
			if (event.type === "auto_retry_start") {
				retries.push({ attempt: event.attempt, errorMessage: event.errorMessage });
			}
			if (event.type === "message_end" && event.message?.role === "assistant") {
				const m = event.message;
				msg.text = (m.content ?? [])
					.filter((c: any) => c.type === "text")
					.map((c: any) => c.text)
					.join("");
				msg.thinking = (m.content ?? [])
					.filter((c: any) => c.type === "thinking")
					.map((c: any) => c.thinking)
					.join("\n\n");
				msg.usage = m.usage;
				msg.stopReason = m.stopReason;
				msg.errorMessage = m.errorMessage;
				msg.model = m.model;
				msg.responseModel = m.responseModel ?? m.model;
				msg.durationMs = m.durationMs;
				usageSum.input += m.usage?.input ?? 0;
				usageSum.output += m.usage?.output ?? 0;
				usageSum.cacheRead += m.usage?.cacheRead ?? 0;
				usageSum.cacheWrite += m.usage?.cacheWrite ?? 0;
				usageSum.reasoning += m.usage?.reasoning ?? 0;
				usageSum.costUsd += m.usage?.cost?.total ?? 0;
			}
		});

		let aborted = false;
		const wallTimer = setTimeout(() => {
			aborted = true;
			void session?.abort().catch(() => {});
		}, opts.wallMin * 60_000);

		const startedAt = Date.now();
		try {
			await session.prompt(promptText, { images, expandPromptTemplates: false });
		} finally {
			clearTimeout(wallTimer);
		}
		const latencyMs = Date.now() - startedAt;

		if (msg.stopReason === undefined && errors.length === 0) {
			// no assistant message arrived (aborted before first token): leave unrecorded
			// so a resume retries this call; the orphan session file is overwritten.
			console.warn(`[${key}/${condition}/r${rep}] no assistant response (aborted=${aborted})`);
			session.dispose();
			return;
		}

		const rawSessionFile = sessionManager.getSessionFile();
		session.dispose();
		session = undefined;
		let sessionFile: string | undefined;
		if (rawSessionFile && existsSync(rawSessionFile)) {
			const dest = join(callDir, `${sessionName}.jsonl`);
			if (resolve(rawSessionFile) !== resolve(dest)) {
				try {
					renameSync(rawSessionFile, dest);
				} catch {
					sessionFile = rel(rawSessionFile);
				}
			}
			if (!sessionFile) sessionFile = rel(dest);
		}

		const error =
			msg.errorMessage ??
			(msg.stopReason === "aborted" ? `aborted (wall ${opts.wallMin} min)` : undefined) ??
			(errors.length > 0 ? errors.join("; ") : undefined);
		const ok = !error && msg.stopReason !== "error";

		if (capture.seen && !shared.providerSystem[condition]) {
			shared.providerSystem[condition] = capture.system;
			const shaped = await shapeWithShim(opts.authShim, capture.payload);
			shared.providerSystemShaped[condition] = extractSystemText(shaped);
		}

		const line = {
			run: opts.run,
			key,
			case: row.case,
			histotype: row.histotype,
			sequence: row.sequence,
			version: row.version,
			condition,
			rep,
			model: opts.model,
			response_model: msg.responseModel ?? msg.model,
			thinking: opts.thinking,
			prompt_template: join("harness/study/prompts", USER_TEMPLATE),
			prompt_sha256: sha256File(join(PROMPTS_DIR, USER_TEMPLATE)),
			system_template: join("harness/study/prompts", spec.systemTemplate),
			system_template_sha256: sha256File(join(PROMPTS_DIR, spec.systemTemplate)),
			system_prompt_sha256: sha256Text(shared.systemPrompts[condition]),
			prompt_text: promptText,
			images: row.images,
			session_file: sessionFile,
			raw_text: msg.text ?? "",
			thinking_text: msg.thinking ?? "",
			latency_ms: latencyMs,
			provider_ms: msg.durationMs,
			usage: {
				input: usageSum.input,
				output: usageSum.output,
				cache_read: usageSum.cacheRead,
				cache_write: usageSum.cacheWrite,
				reasoning: usageSum.reasoning,
				cost_usd: Number(usageSum.costUsd.toFixed(6)),
			},
			stop_reason: msg.stopReason,
			transport_retries: retries,
			ok,
			error: error ?? null,
		};
		appendFileSync(join(runDir, "responses.jsonl"), JSON.stringify(line) + "\n");
		console.log(
			`[${key}/${condition}/r${rep}] ok=${ok} stop=${msg.stopReason} ` +
				`in=${usageSum.input} out=${usageSum.output} ${latencyMs}ms`,
		);
	} catch (err) {
		// Transport-level failure after SDK retries: no response line, so a resume
		// retries this call. Never retried within this process (no re-call here).
		console.error(`[${key}/${condition}/r${rep}] failed: ${(err as Error).message}`);
		appendFileSync(join(runDir, "failures.log"), `${new Date().toISOString()} ${key}/${condition}/r${rep}: ${(err as Error).message}\n`);
		try {
			session?.dispose();
		} catch {
			// ignore
		}
	}
}

// ---------------------------------------------------------------------------
// Shared / main
// ---------------------------------------------------------------------------
interface Shared {
	modelRuntime: ModelRuntimeType;
	systemPrompts: Record<string, string>; // condition -> rendered system prompt
	providerSystem: Record<string, string | undefined>;
	providerSystemShaped: Record<string, string | undefined>;
}

async function main(): Promise<void> {
	const opts = parseArgs(process.argv.slice(2));
	const runDir = join(PROJECT_ROOT, "results", "candidate_replay", opts.run);
	mkdirSync(runDir, { recursive: true });

	if (!existsSync(MANIFEST_PATH)) {
		throw new Error(`no candidate replay manifest at ${MANIFEST_PATH} — run prepare_candidate_replay.py first`);
	}
	const rows: ManifestRow[] = (readFileSync(MANIFEST_PATH, "utf8").trim().split("\n"))
		.filter(Boolean)
		.map((line) => JSON.parse(line));
	const rowsByKey = new Map(rows.map((r) => [r.key, r]));
	let selected = rows;
	if (opts.keys.length > 0) {
		selected = opts.keys.map((k) => {
			const row = rowsByKey.get(k);
			if (!row) throw new Error(`unknown manifest key: ${k}`);
			return row;
		});
	}

	const shimPkg = join(opts.authShim, "package.json");
	const shimVersion = existsSync(shimPkg)
		? (JSON.parse(readFileSync(shimPkg, "utf8")).version as string)
		: undefined;

	// Prompt hashes go into meta.json even when a condition is not run.
	const promptHashes: Record<string, { path: string; sha256: string }> = {};
	for (const file of [USER_TEMPLATE, ...ALL_CONDITIONS.map((c) => CONDITION_SPEC[c].systemTemplate)]) {
		const p = join(PROMPTS_DIR, file);
		if (existsSync(p)) promptHashes[file.replace(".md", "")] = { path: rel(p), sha256: sha256File(p) };
	}
	// Rendered system prompts ({{max_tool_calls}}/{{max_model_prompts}} substituted).
	const systemPrompts: Record<string, string> = {};
	const systemPromptShas: Record<string, string> = {};
	for (const cond of opts.conditions) {
		const template = readFileSync(join(PROMPTS_DIR, CONDITION_SPEC[cond].systemTemplate), "utf8");
		const rendered = renderTemplate(template, SYSTEM_VARS);
		systemPrompts[cond] = rendered;
		systemPromptShas[cond] = sha256Text(rendered);
	}

	// Resumability: keys already recorded are skipped.
	const respPath = join(runDir, "responses.jsonl");
	const done = new Set<string>();
	let priorLines = 0;
	if (existsSync(respPath)) {
		for (const line of readFileSync(respPath, "utf8").split("\n")) {
			if (!line.trim()) continue;
			try {
				const rec = JSON.parse(line);
				done.add(`${rec.key}|${rec.condition}|${rec.rep}`);
				priorLines += 1;
			} catch {
				console.warn(`[resume] skipping malformed response line: ${line.slice(0, 80)}`);
			}
		}
	}

	const calls: CallKey[] = [];
	for (const row of selected) {
		for (const condition of opts.conditions) {
			for (let rep = 0; rep < opts.reps; rep++) {
				if (!done.has(`${row.key}|${condition}|${rep}`)) calls.push({ key: row.key, condition, rep });
			}
		}
	}

	const sdkPkgPath = join(HERE, "node_modules", "@earendil-works", "pi-coding-agent", "package.json");
	const sdkPackageVersion = existsSync(sdkPkgPath)
		? (JSON.parse(readFileSync(sdkPkgPath, "utf8")).version as string)
		: undefined;

	const git = (cmd: string): string | null => {
		try {
			return execSync(cmd, { cwd: PROJECT_ROOT, stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
		} catch {
			return null;
		}
	};
	const meta = {
		run: opts.run,
		git_commit: git("git rev-parse HEAD"),
		git_describe: git("git describe --tags --always --dirty"),
		git_dirty_files: (git("git status --porcelain -- harness eval/v2") ?? "").split("\n").filter((l) => l.trim()),
		stage: "candidate_replay_decide",
		study: "E16 candidate-decision replay",
		created_at: new Date().toISOString(),
		model: opts.model,
		thinking: opts.thinking,
		conditions: opts.conditions,
		condition_system_templates: Object.fromEntries(
			opts.conditions.map((c) => [c, join("harness/study/prompts", CONDITION_SPEC[c].systemTemplate)]),
		),
		reps: opts.reps,
		concurrency: opts.concurrency,
		wall_min_per_call: opts.wallMin,
		tools: "none (tools: []); single tool-less turn, decisions parsed in Python",
		compaction: "off",
		transport_retries: "SDK auto-retry, max 2; parse failures are never retried",
		pi_sdk_version: VERSION,
		pi_sdk_package_version: sdkPackageVersion,
		pi_sdk_version_note:
			"VERSION (export) follows the installed pi CLI release; pi_sdk_package_version is the exact @earendil-works/pi-coding-agent pin in harness/study/package.json",
		auth_shim_path: resolve(opts.authShim),
		auth_shim_version: shimVersion,
		manifest: rel(MANIFEST_PATH),
		manifest_rows: selected.length,
		gt_fields_never_sent: ["precision", "label", "inter_vox", "region_vox", "overlap_mask_vox", "dice_reviewed", "gt_vox", "mask_gt_vox", "agent_accepted"],
		prompts: promptHashes,
		system_prompt_rendered_sha256: systemPromptShas,
		system_prompt_substitutions: SYSTEM_VARS,
		planned_calls: calls.length + priorLines,
		resumed_from_lines: priorLines,
	};
	writeFileSync(join(runDir, "meta.json"), JSON.stringify(meta, null, 2) + "\n");

	const modelRuntime = await ModelRuntime.create({ authPath: join(getAgentDir(), "auth.json") });
	const [provider, ...modelParts] = opts.model.split("/");
	if (!modelRuntime.getModel(provider, modelParts.join("/"))) {
		throw new Error(`model not found: ${opts.model}`);
	}
	const shared: Shared = { modelRuntime, systemPrompts, providerSystem: {}, providerSystemShaped: {} };

	console.log(
		`[replay] run=${opts.run} conditions=${opts.conditions.join(",")} reps=${opts.reps} ` +
			`keys=${selected.length} todo=${calls.length} (done ${priorLines}) concurrency=${opts.concurrency}`,
	);
	console.log(`[replay] pi_sdk=${VERSION} shim=${shimVersion} model=${opts.model} thinking=${opts.thinking}`);

	const queue = [...calls];
	const worker = async (): Promise<void> => {
		for (;;) {
			const callKey = queue.shift();
			if (!callKey) return;
			const row = rowsByKey.get(callKey.key)!;
			await runCall(callKey, row, opts, shared);
		}
	};
	const workerCount = Math.max(1, Math.min(opts.concurrency, queue.length));
	await Promise.all(Array.from({ length: workerCount }, () => worker()));

	// Finalise meta with provenance captured during the run (carry forward the
	// provider-payload captures from a previous invocation when this one made no calls).
	const recordedLines = existsSync(respPath)
		? readFileSync(respPath, "utf8").split("\n").filter((l) => l.trim()).length
		: 0;
	let priorMeta: any = {};
	const metaPath = join(runDir, "meta.json");
	if (existsSync(metaPath)) {
		try {
			priorMeta = JSON.parse(readFileSync(metaPath, "utf8"));
		} catch {
			priorMeta = {};
		}
	}
	const providerSystem: Record<string, string | undefined> = { ...(priorMeta.system_prompt_sent_to_provider ?? {}), ...shared.providerSystem };
	const providerSystemShaped: Record<string, string | undefined> = { ...(priorMeta.system_prompt_shaped_by_auth_shim ?? {}), ...shared.providerSystemShaped };
	const finalMeta = {
		...meta,
		finished_at: new Date().toISOString(),
		recorded_lines: recordedLines,
		system_prompt_sent_to_provider: providerSystem,
		system_prompt_sent_to_provider_sha256: Object.fromEntries(
			Object.entries(providerSystem).map(([c, s]) => [c, s ? sha256Text(s) : undefined]),
		),
		system_prompt_shaped_by_auth_shim: providerSystemShaped,
		system_prompt_notes:
			"captured at pi's before_provider_request hook (pre-shim); the shaped field applies @gotgenes/pi-anthropic-auth request shaping, which the shim performs after that hook",
	};
	writeFileSync(metaPath, JSON.stringify(finalMeta, null, 2) + "\n");
	console.log(`[replay] wrote ${respPath}`);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});
