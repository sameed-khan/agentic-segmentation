#!/usr/bin/env -S npx tsx
/**
 * Phase 5 Study B replay for FRONTIER models (replay_candidates.ts pattern).
 *
 * Same manifests as the MedGemma replay (harness/study/build_decision_sets.py ->
 * results/phase5/studyB/<set>/manifest.jsonl) and the SAME user-side prompt template
 * (prompts/decision_<set>.md) so local and frontier models answer identical
 * decisions. One shared ModelRuntime (subscription auth from ~/.pi/agent/auth.json,
 * REQUIRED auth shim @gotgenes/pi-anthropic-auth), fresh AgentSession per call,
 * isolated resources, compaction off, ZERO tools, no agent loop.
 *
 * Deliberate difference to replay_candidates.ts: a MINIMAL NEUTRAL system prompt
 * (MINIMAL_NEUTRAL_SYSTEM_PROMPT) instead of a study prompt. The SDK default must be
 * overridden EXPLICITLY: resourceLoaderOptions.systemPrompt undefined/empty falls back
 * to pi's discovered coding-agent prompt ("You are an expert coding assistant operating
 * inside pi…"), which would contaminate the decisions. The override uses pi's supported
 * forced-prompt mechanism (a before_agent_start handler returning `systemPrompt`), so the
 * request head is EXACTLY the constant — no generated cwd/tools/docs sections. MedGemma
 * has no system instruction (technical report App. B); the residual difference is
 * disclosed in meta.json together with the exact text sent (captured pre-shim) and the
 * auth-shim-shaped text (the @gotgenes/pi-anthropic-auth OAuth shim ADDS an
 * x-anthropic-billing-header identity/billing system block). Strict JSON parsing
 * happens in Python (harness/study/score_decision.py), never here.
 *
 * Manifest image paths are relative to the SOURCE tree the design runs came from
 * (e.g. /home/ubuntu/agentic-segmentation); pass --images-root pointing there.
 *
 * Output: results/phase5/studyB/<set>/<run>/responses.jsonl (resumable: ok keys are
 * skipped, error rows dropped and retried), meta.json, sessions/.
 *
 * Run:  npx tsx replay_decisions.ts --set done_check --run sonnet-high \
 *           --images-root /home/ubuntu/agentic-segmentation [--model anthropic/claude-opus-5-5]
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
import type { ImageContent } from "@earendil-works/pi-ai";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(HERE, "..", "..");
const STUDY_B_ROOT = join(PROJECT_ROOT, "results", "phase5", "studyB");
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

// Exact target strings — keep in sync with replay_local.py HISTOTYPE_TEXT.
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
const SETS = ["candidate_judgement", "done_check", "wrong_structure"] as const;
// set -> user prompt template (shared with replay_local.py / the MedGemma replay)
const SET_TEMPLATE: Record<(typeof SETS)[number], string> = {
	candidate_judgement: "decision_candidate.md",
	done_check: "decision_done.md",
	wrong_structure: "decision_wrong_structure.md",
};

/**
 * Deliberately minimal, neutral system prompt (review P1-8).
 *
 * `systemPrompt: undefined` does NOT suppress the system prompt: the SDK resolves its
 * default/discovered coding-agent prompt ("You are an expert coding assistant operating
 * inside pi…"), and an empty string is falsy and falls back too. Two supported override
 * mechanisms are combined below:
 * - resourceLoaderOptions.systemPrompt: a non-empty string replaces the generated prompt
 *   (customPrompt preamble path in system-prompt.js) so no pi-generated sections are even
 *   persisted in the session transcript;
 * - a before_agent_start handler returning `systemPrompt` (forceSystemPrompt) projects
 *   EXACTLY this constant as the request's leading system message — the loader's custom
 *   prompt alone would still append a generated `<cwd>` section.
 * The exact text that reaches the provider is asserted per call and persisted in meta.json.
 */
const MINIMAL_NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant.";

interface ManifestRow {
	key: string;
	case: string;
	histotype: string;
	sequence: string;
	set: string;
	source_run?: string;
	version?: string;
	images: string[];
	images_abs?: string[];
	text: string;
	candidate?: { id: string };
}

interface Options {
	set: string;
	run: string;
	reps: number;
	keys: string[];
	concurrency: number;
	model: string;
	thinking: string;
	wallMin: number;
	authShim: string;
	imagesRoot: string;
}

function parseArgs(argv: string[]): Options {
	const opts: Options = {
		set: "",
		run: "",
		reps: 1,
		keys: [],
		concurrency: 4,
		model: "anthropic/claude-sonnet-5-5",
		thinking: "medium",
		wallMin: 5,
		authShim: process.env.AUTH_SHIM ?? DEFAULT_AUTH_SHIM,
		imagesRoot: "",
	};
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		const next = (): string => {
			const value = argv[++i];
			if (value === undefined) throw new Error(`missing value for ${arg}`);
			return value;
		};
		switch (arg) {
			case "--set":
				opts.set = next();
				break;
			case "--run":
				opts.run = next();
				break;
			case "--reps":
				opts.reps = Number(next());
				break;
			case "--keys":
				opts.keys = next().split(",").map((k) => k.trim()).filter(Boolean);
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
			case "--images-root":
				opts.imagesRoot = resolve(next());
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
	if (!opts.set || !SETS.includes(opts.set as any)) {
		throw new Error(`--set is required (one of ${SETS.join(", ")})`);
	}
	if (!opts.run) throw new Error("--run <name> is required");
	if (opts.reps < 1) throw Error("--reps must be >= 1");
	return opts;
}

function printHelp(): void {
	console.log(`Usage: npx tsx replay_decisions.ts --set <name> --run <name> [options]

  --set <name>              one of ${SETS.join(", ")} (manifest under results/phase5/studyB/<set>/)
  --run <name>              run directory under results/phase5/studyB/<set>/
  --images-root <dir>       source tree root the manifest image paths resolve against
                            (e.g. /home/ubuntu/agentic-segmentation; defaults to images_abs)
  --reps <n>                repeats per key (default 1)
  --keys K1,K2              subset of manifest keys (default: all)
  --concurrency <n>         calls in flight in this process (default 4)
  --model <provider/id>     (default anthropic/claude-sonnet-5-5)
  --thinking <level>        (default medium)
  --wall-min <n>            per-call wall-time abort (default 5)
  --auth-shim <path>        default ~/.pi/agent/npm/node_modules/@gotgenes/pi-anthropic-auth`);
}

// ---------------------------------------------------------------------------
// Helpers (same as replay_candidates.ts)
// ---------------------------------------------------------------------------
function sha256File(path: string): string {
	return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function sha256Text(text: string): string {
	return createHash("sha256").update(text).digest("hex");
}

function renderTemplate(text: string, vars: Record<string, string>): string {
	return text.replace(/\{\{(\w+)\}\}/g, (match, key: string) =>
		Object.prototype.hasOwnProperty.call(vars, key) ? vars[key] : match,
	);
}

function rel(path: string): string {
	return relative(PROJECT_ROOT, resolve(path));
}

function imageContent(row: ManifestRow, index: number, opts: Options): ImageContent {
	const relPath = row.images[index];
	const abs = row.images_abs?.[index];
	const path =
		abs && existsSync(abs)
			? abs
			: opts.imagesRoot
				? join(opts.imagesRoot, relPath)
				: resolve(PROJECT_ROOT, relPath);
	return { type: "image", data: readFileSync(path).toString("base64"), mimeType: "image/png" };
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

/** Best-effort reproduction of the shim's OAuth payload shaping (see replay_candidates.ts). */
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
async function runCall(
	row: ManifestRow,
	rep: number,
	opts: Options,
	shared: Shared,
): Promise<void> {
	const runDir = join(STUDY_B_ROOT, opts.set, opts.run);
	const callDir = join(runDir, "sessions");
	mkdirSync(callDir, { recursive: true });
	const sessionName = `${row.key.replace(/[/:]/g, "_")}_r${rep}`;

	const templatePath = join(PROMPTS_DIR, SET_TEMPLATE[opts.set as (typeof SETS)[number]]);
	const template = readFileSync(templatePath, "utf8");
	const promptText = renderTemplate(template, {
		text: row.text ?? "",
		case: row.case,
		target: HISTOTYPE_TEXT[row.histotype] ?? row.histotype,
		sequence_text: SEQUENCE_TEXT[row.sequence] ?? row.sequence,
		candidate_id: row.candidate?.id ?? "",
	});
	const images = row.images.map((_, i) => imageContent(row, i, opts));

	const capture: { seen: boolean; payload?: unknown; system?: string } = { seen: false };
	const captureFactory: ExtensionFactory = (pi: ExtensionAPI) => {
		pi.on("before_provider_request", (event) => {
			capture.seen = true;
			capture.payload = event.payload;
			capture.system = extractSystemText(event.payload) ?? capture.system;
		});
		// Force the minimal neutral prompt as the request's leading system message
		// (supported mechanism: a before_agent_start handler returning `systemPrompt`
		// sets forceSystemPrompt, which projects EXACTLY this text with no additional
		// prompt sections; review P1-8).
		pi.on("before_agent_start", () => ({ systemPrompt: MINIMAL_NEUTRAL_SYSTEM_PROMPT }));
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
				// Explicit minimal neutral prompt: undefined would select the SDK's
				// coding-agent default (review P1-8). Asserted against the captured
				// payload below and persisted in meta.json.
				systemPrompt: MINIMAL_NEUTRAL_SYSTEM_PROMPT,
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
			tools: [], // zero tools: single tool-less decision turn
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

		// Never drop a planned call (review P1-4): a missing assistant response is a
		// terminal error row (ok=false), not a silent return.
		const noResponse = msg.stopReason === undefined;
		if (capture.seen && capture.system !== MINIMAL_NEUTRAL_SYSTEM_PROMPT) {
			errors.push(
				`provider system prompt mismatch: got ${JSON.stringify((capture.system ?? "").slice(0, 120))}`,
			);
		}

		const error = noResponse
			? (errors.length > 0 ? errors.join("; ") : `no assistant response (aborted=${aborted})`)
			: (msg.errorMessage ??
				(msg.stopReason === "aborted" ? `aborted (wall ${opts.wallMin} min)` : undefined) ??
				(errors.length > 0 ? errors.join("; ") : undefined));
		const ok = !noResponse && !error && msg.stopReason !== "error";

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

		if (capture.seen && !shared.providerSystem) {
			shared.providerSystem = capture.system;
			const shaped = await shapeWithShim(opts.authShim, capture.payload);
			shared.providerSystemShaped = extractSystemText(shaped);
		}

		const line = {
			run: opts.run,
			key: row.key,
			set: row.set,
			case: row.case,
			histotype: row.histotype,
			sequence: row.sequence,
			source_run: row.source_run,
			version: row.version,
			model: opts.model,
			rep,
			response_model: msg.responseModel ?? msg.model,
			thinking: opts.thinking,
			prompt_template: join("harness/study/prompts", `decision_${opts.set}.md`),
			prompt_sha256: sha256File(templatePath),
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
			stop_reason: noResponse ? null : msg.stopReason,
			transport_retries: retries,
			system_prompt_sha256: sha256Text(MINIMAL_NEUTRAL_SYSTEM_PROMPT),
			ok,
			error: error ?? null,
		};
		appendFileSync(join(runDir, "responses.jsonl"), JSON.stringify(line) + "\n");
		console.log(
			`[${row.key}/r${rep}] ok=${ok} stop=${noResponse ? "none" : msg.stopReason} ` +
				`in=${usageSum.input} out=${usageSum.output} ${latencyMs}ms`,
		);
	} catch (err) {
		console.error(`[${row.key}/r${rep}] failed: ${(err as Error).message}`);
		appendFileSync(
			join(runDir, "failures.log"),
			`${new Date().toISOString()} ${row.key}/r${rep}: ${(err as Error).message}\n`,
		);
		// Never drop a planned call (review P1-4): the failure also gets a terminal
		// response row (ok=false) so the scorer's coverage validation and denominators
		// see it instead of the call silently vanishing.
		try {
			appendFileSync(
				join(runDir, "responses.jsonl"),
				JSON.stringify({
					run: opts.run,
					key: row.key,
					set: row.set,
					case: row.case,
					histotype: row.histotype,
					sequence: row.sequence,
					source_run: row.source_run ?? null,
					version: row.version ?? null,
					model: opts.model,
					rep,
					thinking: opts.thinking,
					prompt_template: join("harness/study/prompts", `decision_${opts.set}.md`),
					prompt_sha256: sha256File(templatePath),
					prompt_text: promptText,
					images: row.images,
					session_file: null,
					raw_text: "",
					thinking_text: "",
					latency_ms: null,
					provider_ms: null,
					usage: {
						input: 0,
						output: 0,
						cache_read: 0,
						cache_write: 0,
						reasoning: 0,
						cost_usd: 0,
					},
					stop_reason: "error",
					transport_retries: [],
					system_prompt_sha256: sha256Text(MINIMAL_NEUTRAL_SYSTEM_PROMPT),
					ok: false,
					error: (err as Error).message,
				}) + "\n",
			);
		} catch (logErr) {
			console.error(`[${row.key}/r${rep}] could not write terminal row: ${(logErr as Error).message}`);
		}
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
	modelRuntime: any;
	providerSystem?: string;      // system prompt captured pre-shim (before_provider_request)
	providerSystemShaped?: string; // system prompt after @gotgenes/pi-anthropic-auth shaping
}

async function main(): Promise<void> {
	const opts = parseArgs(process.argv.slice(2));
	const setDir = join(STUDY_B_ROOT, opts.set);
	const manifestPath = join(setDir, "manifest.jsonl");
	if (!existsSync(manifestPath)) {
		throw new Error(`no decision-set manifest at ${manifestPath} — run build_decision_sets.py first`);
	}
	const templatePath = join(PROMPTS_DIR, SET_TEMPLATE[opts.set as (typeof SETS)[number]]);
	if (!existsSync(templatePath)) {
		throw new Error(`no decision prompt template: ${templatePath}`);
	}
	const rows: ManifestRow[] = readFileSync(manifestPath, "utf8").trim().split("\n")
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

	const runDir = join(setDir, opts.run);
	mkdirSync(runDir, { recursive: true });
	const shimPkg = join(opts.authShim, "package.json");
	const shimVersion = existsSync(shimPkg)
		? (JSON.parse(readFileSync(shimPkg, "utf8")).version as string)
		: undefined;

	// Resumability (ask.ts convention): ok rows are skipped; error rows dropped so a
	// rerun with the same --run retries them.
	const respPath = join(runDir, "responses.jsonl");
	const done = new Set<string>();
	let keptLines: string[] = [];
	let priorErrorLines = 0;
	if (existsSync(respPath)) {
		for (const line of readFileSync(respPath, "utf8").split("\n")) {
			if (!line.trim()) continue;
			try {
				const rec = JSON.parse(line);
				if (rec.ok === true) {
					done.add(`${rec.key}|${rec.rep}`);
					keptLines.push(line);
				} else {
					priorErrorLines += 1;
				}
			} catch {
				console.warn(`[resume] skipping malformed response line: ${line.slice(0, 80)}`);
			}
		}
		if (priorErrorLines > 0) {
			writeFileSync(respPath, keptLines.map((l) => `${l}\n`).join(""));
			console.log(`[resume] dropping ${priorErrorLines} error row(s) for retry`);
		}
	}

	const calls: { row: ManifestRow; rep: number }[] = [];
	for (const row of selected) {
		for (let rep = 0; rep < opts.reps; rep++) {
			if (!done.has(`${row.key}|${rep}`)) calls.push({ row, rep });
		}
	}

	const git = (cmd: string): string | null => {
		try {
			return execSync(cmd, { cwd: PROJECT_ROOT, stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
		} catch {
			return null;
		}
	};
	const meta = {
		run: opts.run,
		set: opts.set,
		git_commit: git("git rev-parse HEAD"),
		git_describe: git("git describe --tags --always --dirty"),
		stage: "phase5_studyB_decision_replay_frontier",
		created_at: new Date().toISOString(),
		model: opts.model,
		thinking: opts.thinking,
		reps: opts.reps,
		concurrency: opts.concurrency,
		wall_min_per_call: opts.wallMin,
		tools: "none (tools: []); single tool-less turn, decisions parsed in Python",
		compaction: "off",
		system_prompt: MINIMAL_NEUTRAL_SYSTEM_PROMPT,
		system_prompt_note:
			"deliberate minimal neutral prompt; the SDK default (a coding-agent prompt) is " +
			"selected whenever resourceLoaderOptions.systemPrompt is undefined/empty, so it is " +
			"overridden explicitly and the effective payload is asserted per call",
		transport_retries: "SDK auto-retry, max 2; parse failures are never retried",
		pi_sdk_version: VERSION,
		auth_shim_path: resolve(opts.authShim),
		auth_shim_version: shimVersion,
		manifest: rel(manifestPath),
		manifest_sha256: sha256File(manifestPath),
		manifest_rows: selected.length,
		selected_keys: selected.map((r) => r.key),
		images_root: opts.imagesRoot || null,
		prompt_template: rel(templatePath),
		prompt_sha256: sha256File(templatePath),
		gt_fields_never_sent: ["label", "precision", "dice_state", "dice_best", "version_dice_gain",
			"oracle_dice_gain", "region_vox", "inter_vox", "dice_reviewed"],
		planned_calls: calls.length + done.size,
		resumed_from_lines: done.size,
	};
	// Prior meta (only meaningful on resume): used for the immutable-config check and,
	// on a NO-OP resume, to keep the previously captured system-prompt provenance.
	const priorMetaPath = join(runDir, "meta.json");
	const priorMeta: Record<string, unknown> | null =
		done.size > 0 && existsSync(priorMetaPath)
			? (JSON.parse(readFileSync(priorMetaPath, "utf8")) as Record<string, unknown>)
			: null;
	// Immutable-config check (review P2-4): resuming must not mix configuration variants.
	if (priorMeta) {
		const immutableKeys = ["set", "model", "thinking", "reps", "manifest", "manifest_sha256", "selected_keys", "images_root", "prompt_sha256"] as const;
		const drifted = immutableKeys.filter((k) => JSON.stringify(priorMeta[k]) !== JSON.stringify(meta[k]));
		if (drifted.length > 0) {
			throw new Error(
				`run config drifted from prior meta.json (${drifted.join(", ")}); use a new --run name`,
			);
		}
	}
	writeFileSync(join(runDir, "meta.json"), JSON.stringify(meta, null, 2) + "\n");

	const modelRuntime = await ModelRuntime.create({ authPath: join(getAgentDir(), "auth.json") });
	const [provider, ...modelParts] = opts.model.split("/");
	if (!modelRuntime.getModel(provider, modelParts.join("/"))) {
		throw new Error(`model not found: ${opts.model}`);
	}
	const shared: Shared = { modelRuntime };

	console.log(
		`[replay] set=${opts.set} run=${opts.run} model=${opts.model} rows=${selected.length} ` +
			`todo=${calls.length} (done ${done.size}) concurrency=${opts.concurrency}`,
	);
	const queue = [...calls];
	const worker = async (): Promise<void> => {
		for (;;) {
			const call = queue.shift();
			if (!call) return;
			await runCall(call.row, call.rep, opts, shared);
		}
	};
	const workerCount = Math.max(1, Math.min(opts.concurrency, queue.length));
	await Promise.all(Array.from({ length: workerCount }, () => worker()));

	const recordedLines = existsSync(respPath)
		? readFileSync(respPath, "utf8").split("\n").filter((l) => l.trim()).length
		: 0;
	// No-op resume (every call already complete): no request runs, so no capture happens
	// and shared.providerSystem stays unset — fall back to the PRIOR captured values
	// instead of overwriting the provenance with null (review fix round).
	const priorProviderSystem =
		typeof priorMeta?.system_prompt_sent_to_provider === "string"
			? (priorMeta.system_prompt_sent_to_provider as string)
			: null;
	const priorProviderSystemShaped =
		typeof priorMeta?.system_prompt_shaped_by_auth_shim === "string"
			? (priorMeta.system_prompt_shaped_by_auth_shim as string)
			: null;
	const providerSystem = shared.providerSystem ?? priorProviderSystem;
	const providerSystemShaped = shared.providerSystemShaped ?? priorProviderSystemShaped;
	const finalMeta = {
		...meta,
		finished_at: new Date().toISOString(),
		recorded_lines: recordedLines,
		system_prompt_sent_to_provider: providerSystem,
		system_prompt_sent_to_provider_sha256:
			typeof providerSystem === "string" ? sha256Text(providerSystem) : null,
		system_prompt_shaped_by_auth_shim: providerSystemShaped,
		system_prompt_shaped_by_auth_shim_sha256:
			typeof providerSystemShaped === "string" ? sha256Text(providerSystemShaped) : null,
		system_prompt_provider_notes:
			"captured at pi's before_provider_request hook (pre-shim); the shaped field applies " +
			"@gotgenes/pi-anthropic-auth request shaping, which the shim performs after that hook. " +
			"The Anthropic OAuth shim ADDS an x-anthropic-billing-header identity/billing system " +
			"block (identifying the request as Claude Code usage) and may rewrite the Pi identity; " +
			"the shaped text is what the provider actually receives.",
	};
	writeFileSync(join(runDir, "meta.json"), JSON.stringify(finalMeta, null, 2) + "\n");
	console.log(`[replay] wrote ${respPath}`);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});
