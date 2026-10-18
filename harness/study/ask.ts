#!/usr/bin/env -S npx tsx
/**
 * Phase 2 localization study, stage 2: single-turn model calls (design §3 Phase 2,
 * revision 2026-10-18: runs on the 30 design cases; research/HANDOFF_2026-10-18.md §4.3).
 *
 * Same pi SDK plumbing as harness/runner/run_cases.ts — one shared ModelRuntime
 * (subscription auth from ~/.pi/agent/auth.json, REQUIRED auth shim
 * @gotgenes/pi-anthropic-auth), fresh AgentSession per call, isolated resources
 * (no skills/context files/prompt templates/discovered extensions), compaction off,
 * ZERO tools — but no agent loop: one user message (prompt text + inline PNG(s)),
 * one reply, recorded raw.
 *
 * Prompts live in prompts/*.md (sha256 recorded in the run meta.json). Images come
 * from the prepare stage manifest (results/phase2_dev/prepare/manifest.jsonl).
 *
 * Output: results/phase2_dev/<run>/responses.jsonl (one line per case×condition×rep,
 * resumable: completed keys are skipped), meta.json, sessions/<case>_<cond>_r<rep>.jsonl.
 *
 * Run:  npx tsx ask.ts --run phase2-dev-01 [--conditions R,S,S-gen] [--reps 3]
 *                       [--cases F14,M07] [--concurrency 4]
 */
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
const PREPARE_DIR = join(PROJECT_ROOT, "results", "phase2_dev", "prepare");
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

// Exact target strings — keep in sync with segd/data.py HISTOTYPE_TEXT and
// harness/runner/run_cases.ts (the runner's task messages use these).
const HISTOTYPE_TEXT: Record<string, string> = {
	F: "fibrous soft-tissue tumor",
	L: "lipomatous soft-tissue tumor",
	M: "myxoid soft-tissue tumor",
	N: "peripheral nerve sheath tumor",
	V: "vascular soft-tissue tumor",
};
const SEQUENCE_TEXT: Record<string, string> = {
	T1: "T1-weighted",
	T2: "T2-weighted fat-suppressed",
};
const S_GEN_TARGET = "soft-tissue mass";
const ALL_CONDITIONS = ["R", "S", "S-gen", "S-desc", "S-ex"] as const;

// condition -> prompt template file + which images go into the message (order matters)
const CONDITION_SPEC: Record<string, { template: string; images: (row: ManifestRow) => string[] }> = {
	R: { template: "r.md", images: (row) => [row.image_R] },
	S: { template: "s.md", images: (row) => [row.image_S] },
	"S-gen": { template: "s.md", images: (row) => [row.image_S] },
	"S-desc": { template: "s_desc.md", images: (row) => [row.image_S] },
	"S-ex": { template: "s_ex.md", images: (row) => [row.image_example, row.image_S].filter(Boolean) as string[] },
};

// ---------------------------------------------------------------------------
// Types / options
// ---------------------------------------------------------------------------
interface ManifestRow {
	case: string;
	histotype: string;
	sequence: string;
	image_S: string;
	image_R: string;
	image_example: string | null;
}

interface Options {
	conditions: string[];
	reps: number;
	run: string;
	cases: string[];
	concurrency: number;
	model: string;
	thinking: string;
	wallMin: number;
	authShim: string;
	prepareDir: string;
}

function parseArgs(argv: string[]): Options {
	const opts: Options = {
		conditions: ["R", "S", "S-gen"],
		reps: 3,
		run: "",
		cases: [],
		concurrency: 4,
		model: "anthropic/claude-sonnet-5-5",
		thinking: "medium",
		wallMin: 5,
		authShim: process.env.AUTH_SHIM ?? DEFAULT_AUTH_SHIM,
		prepareDir: PREPARE_DIR,
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
			case "--cases":
				opts.cases = next().split(",").map((c) => c.trim()).filter(Boolean);
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
			case "--prepare-dir":
				opts.prepareDir = resolve(next());
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
	console.log(`Usage: npx tsx ask.ts --run <name> [options]

  --run <name>              run directory under results/phase2_dev/ (required)
  --conditions R,S,S-gen    comma list from ${ALL_CONDITIONS.join(", ")} (default R,S,S-gen)
  --reps <n>                repeats per case x condition (default 3)
  --cases F14,M07           subset of design cases (default: all in the prepare manifest)
  --concurrency <n>         calls in flight in this process (default 4)
  --model <provider/id>     (default anthropic/claude-sonnet-5-5)
  --thinking <level>        (default medium)
  --wall-min <n>            per-call wall-time abort (default 5)
  --auth-shim <path>        default ~/.pi/agent/npm/node_modules/@gotgenes/pi-anthropic-auth
  --prepare-dir <path>      default results/phase2_dev/prepare`);
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
	case: string;
	condition: string;
	rep: number;
}

async function runCall(
	key: CallKey,
	row: ManifestRow,
	opts: Options,
	shared: Shared,
): Promise<void> {
	const { case: caseId, condition, rep } = key;
	const spec = CONDITION_SPEC[condition];
	const target = condition === "S-gen" ? S_GEN_TARGET : HISTOTYPE_TEXT[row.histotype] ?? row.histotype;
	const vars: Record<string, string> = {
		target,
		sequence_text: SEQUENCE_TEXT[row.sequence] ?? row.sequence,
		description: "",
	};
	if (condition === "S-desc") {
		const desc = shared.descriptions?.[row.histotype]?.text;
		if (!desc) throw new Error(`no description for histotype ${row.histotype} in descriptions.json`);
		vars.description = desc;
	}
	const template = readFileSync(join(PROMPTS_DIR, spec.template), "utf8");
	const promptText = renderTemplate(template, vars);
	const images = spec.images(row).map(imageContent);

	const runDir = join(PROJECT_ROOT, "results", "phase2_dev", opts.run);
	const callDir = join(runDir, "sessions");
	mkdirSync(callDir, { recursive: true });
	const sessionName = `${caseId}_${condition}_r${rep}`;

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
				systemPrompt: shared.systemPrompt,
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
			tools: [], // zero tools: this study has no action space
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
			console.warn(`[${caseId}/${condition}/r${rep}] no assistant response (aborted=${aborted})`);
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

		if (capture.seen && !shared.providerSystem) {
			shared.providerSystem = capture.system;
			const shaped = await shapeWithShim(opts.authShim, capture.payload);
			shared.providerSystemShaped = extractSystemText(shaped);
		}

		const line = {
			run: opts.run,
			case: caseId,
			histotype: row.histotype,
			sequence: row.sequence,
			condition,
			rep,
			model: opts.model,
			response_model: msg.responseModel ?? msg.model,
			thinking: opts.thinking,
			prompt_template: join("harness/study/prompts", spec.template),
			prompt_sha256: sha256File(join(PROMPTS_DIR, spec.template)),
			prompt_text: promptText,
			images: spec.images(row),
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
			`[${caseId}/${condition}/r${rep}] ok=${ok} stop=${msg.stopReason} ` +
				`in=${usageSum.input} out=${usageSum.output} ${latencyMs}ms`,
		);
	} catch (err) {
		// Transport-level failure after SDK retries: no response line, so a resume
		// retries this call. Never retried within this process (no re-call here).
		console.error(`[${caseId}/${condition}/r${rep}] failed: ${(err as Error).message}`);
		appendFileSync(join(runDir, "failures.log"), `${new Date().toISOString()} ${caseId}/${condition}/r${rep}: ${(err as Error).message}\n`);
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
	systemPrompt: string;
	systemPromptSha: string;
	shimVersion?: string;
	descriptions?: Record<string, { text: string }>;
	providerSystem?: string;
	providerSystemShaped?: string;
}

async function main(): Promise<void> {
	const opts = parseArgs(process.argv.slice(2));
	const runDir = join(PROJECT_ROOT, "results", "phase2_dev", opts.run);
	mkdirSync(runDir, { recursive: true });

	const manifestPath = join(opts.prepareDir, "manifest.jsonl");
	if (!existsSync(manifestPath)) {
		throw new Error(`no prepare manifest at ${manifestPath} — run prepare.py first`);
	}
	const rows: ManifestRow[] = (readFileSync(manifestPath, "utf8").trim().split("\n"))
		.filter(Boolean)
		.map((line) => JSON.parse(line));
	const rowsByCase = new Map(rows.map((r) => [r.case, r]));
	let selected = rows;
	if (opts.cases.length > 0) {
		selected = rows.filter((r) => opts.cases.includes(r.case));
		const missing = opts.cases.filter((c) => !rowsByCase.has(c));
		if (missing.length > 0) throw new Error(`unknown cases: ${missing.join(", ")}`);
	}
	if (opts.conditions.includes("S-desc") || opts.conditions.includes("S-ex")) {
		const noEx = selected.filter((r) => !r.image_example);
		if (noEx.length > 0) {
			throw new Error(`no example image for: ${noEx.map((r) => r.case).join(", ")} — rerun prepare.py`);
		}
	}

	const descriptionsPath = join(HERE, "descriptions.json");
	const descriptions = existsSync(descriptionsPath)
		? (JSON.parse(readFileSync(descriptionsPath, "utf8")).descriptions as Record<string, { text: string }>)
		: undefined;

	const shimPkg = join(opts.authShim, "package.json");
	const shimVersion = existsSync(shimPkg)
		? (JSON.parse(readFileSync(shimPkg, "utf8")).version as string)
		: undefined;

	// Template hashes go into meta.json even when a condition is not run.
	const promptHashes: Record<string, { path: string; sha256: string }> = {};
	for (const file of ["system.md", "r.md", "s.md", "s_desc.md", "s_ex.md"]) {
		const p = join(PROMPTS_DIR, file);
		if (existsSync(p)) promptHashes[file.replace(".md", "")] = { path: rel(p), sha256: sha256File(p) };
	}
	const systemPrompt = readFileSync(join(PROMPTS_DIR, "system.md"), "utf8");

	// Resumability: keys already recorded are skipped.
	const respPath = join(runDir, "responses.jsonl");
	const done = new Set<string>();
	let priorLines = 0;
	if (existsSync(respPath)) {
		for (const line of readFileSync(respPath, "utf8").split("\n")) {
			if (!line.trim()) continue;
			try {
				const rec = JSON.parse(line);
				done.add(`${rec.case}|${rec.condition}|${rec.rep}`);
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
				if (!done.has(`${row.case}|${condition}|${rep}`)) calls.push({ case: row.case, condition, rep });
			}
		}
	}

	const sdkPkgPath = join(HERE, "node_modules", "@earendil-works", "pi-coding-agent", "package.json");
	const sdkPackageVersion = existsSync(sdkPkgPath)
		? (JSON.parse(readFileSync(sdkPkgPath, "utf8")).version as string)
		: undefined;

	const meta = {
		run: opts.run,
		stage: "phase2_localization_ask",
		created_at: new Date().toISOString(),
		model: opts.model,
		thinking: opts.thinking,
		conditions: opts.conditions,
		reps: opts.reps,
		concurrency: opts.concurrency,
		wall_min_per_call: opts.wallMin,
		tools: "none (tools: [])",
		compaction: "off",
		transport_retries: "SDK auto-retry, max 2; parse failures are never retried",
		pi_sdk_version: VERSION,
		pi_sdk_package_version: sdkPackageVersion,
		pi_sdk_version_note:
			"VERSION (export) follows the installed pi CLI release; pi_sdk_package_version is the exact @earendil-works/pi-coding-agent pin in harness/study/package.json",
		auth_shim_path: resolve(opts.authShim),
		auth_shim_version: shimVersion,
		prepare_manifest: rel(manifestPath),
		prompts: promptHashes,
		system_prompt: { path: rel(join(PROMPTS_DIR, "system.md")), sha256: sha256Text(systemPrompt) },
		s_gen_target: S_GEN_TARGET,
		descriptions_file: existsSync(descriptionsPath) ? rel(descriptionsPath) : undefined,
		planned_calls: calls.length + priorLines,
		resumed_from_lines: priorLines,
	};
	writeFileSync(join(runDir, "meta.json"), JSON.stringify(meta, null, 2) + "\n");

	const modelRuntime = await ModelRuntime.create({ authPath: join(getAgentDir(), "auth.json") });
	const [provider, ...modelParts] = opts.model.split("/");
	if (!modelRuntime.getModel(provider, modelParts.join("/"))) {
		throw new Error(`model not found: ${opts.model}`);
	}
	const shared: Shared = {
		modelRuntime,
		systemPrompt,
		systemPromptSha: sha256Text(systemPrompt),
		shimVersion,
		descriptions,
	};

	console.log(
		`[ask] run=${opts.run} conditions=${opts.conditions.join(",")} reps=${opts.reps} ` +
			`cases=${selected.length} todo=${calls.length} (done ${priorLines}) concurrency=${opts.concurrency}`,
	);
	console.log(`[ask] pi_sdk=${VERSION} shim=${shimVersion} model=${opts.model} thinking=${opts.thinking}`);

	const queue = [...calls];
	const worker = async (): Promise<void> => {
		for (;;) {
			const key = queue.shift();
			if (!key) return;
			const row = rowsByCase.get(key.case)!;
			await runCall(key, row, opts, shared);
		}
	};
	const workerCount = Math.max(1, Math.min(opts.concurrency, queue.length));
	await Promise.all(Array.from({ length: workerCount }, () => worker()));

	// Finalise meta with provenance captured during the run (carry forward the
	// provider-payload capture from a previous invocation when this one made no calls).
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
	const providerSystem = shared.providerSystem ?? priorMeta.system_prompt_sent_to_provider;
	const providerSystemShaped = shared.providerSystemShaped ?? priorMeta.system_prompt_shaped_by_auth_shim;
	const finalMeta = {
		...meta,
		finished_at: new Date().toISOString(),
		recorded_lines: recordedLines,
		system_prompt_sent_to_provider: providerSystem,
		system_prompt_sent_to_provider_sha256: providerSystem ? sha256Text(providerSystem) : undefined,
		system_prompt_shaped_by_auth_shim: providerSystemShaped,
		system_prompt_notes:
			"captured at pi's before_provider_request hook (pre-shim); the shaped field applies @gotgenes/pi-anthropic-auth request shaping, which the shim performs after that hook",
	};
	writeFileSync(join(runDir, "meta.json"), JSON.stringify(finalMeta, null, 2) + "\n");
	console.log(`[ask] wrote ${respPath}`);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});
