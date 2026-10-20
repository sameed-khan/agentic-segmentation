/**
 * Command-line parsing and per-dataset default application for run_cases.ts.
 *
 * Own module (like system_prompt.ts) so the explicit-flag tracking and the
 * default-application rules are unit-testable without importing run_cases.ts
 * (whose import runs main()). An option the user passed explicitly is recorded in
 * `explicit` BEFORE the switch's break; dataset defaults apply only to flags that are
 * NOT in that set, independent of the flag's order relative to --dataset. An explicit
 * --task-file always wins — including when its resolved path equals the default.
 */
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { listPresetNames, loadPreset, presetToArgv } from "./presets.ts";
import { DEFAULT_SYSTEM_FILE } from "./system_prompt.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(HERE, "..", "..");
const HARNESS_DIR = join(PROJECT_ROOT, "harness");
export const DEFAULT_EXTENSION_DIR = join(HARNESS_DIR, "extension");
export const DEFAULT_TASK_FILE = join(HARNESS_DIR, "prompts", "task.md");
export const DEFAULT_TASK_MULTI_FILE = join(HARNESS_DIR, "prompts", "task_multi.md");
const DEFAULT_AUTH_SHIM = join(
	homedir(),
	".pi",
	"agent",
	"npm",
	"node_modules",
	"@gotgenes",
	"pi-anthropic-auth",
);

/** Runner-selectable datasets (Phase 4 = BEAMSTER only; 'mstt' = unchanged behaviour). */
export const DATASETS = ["mstt", "beamster"] as const;

/** Interactive segmenter behind the identical tool surface (Phase 6 swap). */
export const SEGMENTERS = ["nninteractive", "sammed3d"] as const;

/** Per-dataset budget/prompt defaults (decision 4): applied for options the user did
 *  NOT pass explicitly. Static per dataset — never derived from GT lesion counts. */
export const DATASET_DEFAULTS: Record<string, { wallMin: number; maxToolCalls: number; maxModelPrompts: number }> = {
	beamster: { wallMin: 12, maxToolCalls: 50, maxModelPrompts: 20 },
};

/** The `v3` object sent in the POST /cases body and recorded in run meta (null = off). */
export interface V3Spec {
	pointer: string | null;
	pointer_url: string | null;
	medgemma: string | null;
	medgemma_url: string | null;
	checkpoints: string[];
}

export interface Options {
	preset: string | null;         // --preset <name>: experiments/presets/<name>.json (provenance only)
	cases: string[];
	split: string | null;
	dataset: string;
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
	segmenter: string;
	segmenterUrl: string | null;
	v3Pointer: string | null;
	v3PointerUrl: string | null;
	v3Medgemma: string | null;
	v3MedgemmaUrl: string | null;
	v3Checkpoints: string[];
	v3: V3Spec | null;
	extension: string;
	dryStub: boolean;
	authShim: string;
	resultsDir: string;
	note: string;
	examples: string;
	systemFile: string;
	systemFileSupplied: boolean;   // tracked apart from the value: an explicit
	taskFile: string;              // default-path override must still win
	resume: boolean;
}

export function applyDatasetProfile(opts: Options, explicit: Set<string>): void {
	const d = DATASET_DEFAULTS[opts.dataset];
	if (d) {
		if (!explicit.has("--wall-min")) opts.wallMin = d.wallMin;
		if (!explicit.has("--max-tool-calls")) opts.maxToolCalls = d.maxToolCalls;
		if (!explicit.has("--max-model-prompts")) opts.maxModelPrompts = d.maxModelPrompts;
	}
	if (!explicit.has("--task-file")) {
		opts.taskFile = opts.dataset === "mstt" ? DEFAULT_TASK_FILE : DEFAULT_TASK_MULTI_FILE;
	}
}

/**
 * `--preset <name>` expands into the preset's recorded flags placed BEFORE the user's own
 * flags (explicit flags win; a user `--cases`/`--split` replaces the preset's selection).
 * Returns the argv to parse and the preset name (null without --preset).
 */
export function expandPreset(argv: string[]): { argv: string[]; preset: string | null } {
	const idx = argv.indexOf("--preset");
	if (idx < 0) return { argv, preset: null };
	if (argv.indexOf("--preset", idx + 1) >= 0) throw new Error("--preset given twice");
	const name = argv[idx + 1];
	if (name === undefined || name.startsWith("--")) throw new Error("missing value for --preset");
	const rest = [...argv.slice(0, idx), ...argv.slice(idx + 2)];
	const userFlags = new Set(rest.filter((t) => t.startsWith("--")));
	return { argv: [...presetToArgv(loadPreset(name), userFlags), ...rest], preset: name };
}

export function parseArgs(rawArgv: string[]): Options {
	const { argv, preset } = expandPreset(rawArgv);
	// options the user passed explicitly (per-dataset defaults must not override them);
	// each flag is added BEFORE its break so the tracking cannot be skipped
	const explicit = new Set<string>();
	const opts: Options = {
		preset,
		cases: [],
		split: null,
		dataset: "mstt",
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
		segmenter: "nninteractive",
		segmenterUrl: null,
		v3Pointer: null,
		v3PointerUrl: null,
		v3Medgemma: null,
		v3MedgemmaUrl: null,
		v3Checkpoints: [],
		v3: null,
		extension: DEFAULT_EXTENSION_DIR,
		dryStub: false,
		authShim: process.env.AUTH_SHIM ?? DEFAULT_AUTH_SHIM,
		resultsDir: join(PROJECT_ROOT, "results"),
		note: "",
		resume: false,
		examples: "",
		systemFile: DEFAULT_SYSTEM_FILE,
		systemFileSupplied: false,
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
			case "--list-presets":
				console.log(listPresetNames().join("\n"));
				process.exit(0);
				break;
			case "--cases":
				opts.cases = next().split(",").map((c) => c.trim()).filter(Boolean);
				break;
			case "--split":
				opts.split = next();
				break;
			case "--dataset":
				opts.dataset = next();
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
				explicit.add("--wall-min");
				break;
			case "--max-nudges":
				opts.maxNudges = Number(next());
				break;
			case "--max-tool-calls":
				opts.maxToolCalls = Number(next());
				explicit.add("--max-tool-calls");
				break;
			case "--max-model-prompts":
				opts.maxModelPrompts = Number(next());
				explicit.add("--max-model-prompts");
				break;
			case "--segd-url":
				opts.segdUrl = next();
				break;
			case "--nni-url":
				opts.nniUrl = next();
				break;
			case "--segmenter":
				opts.segmenter = next();
				break;
			case "--segmenter-url":
				opts.segmenterUrl = next();
				break;
			case "--v3-pointer":
				opts.v3Pointer = next();
				break;
			case "--v3-pointer-url":
				opts.v3PointerUrl = next();
				break;
			case "--v3-medgemma":
				opts.v3Medgemma = next();
				break;
			case "--v3-medgemma-url":
				opts.v3MedgemmaUrl = next();
				break;
			case "--v3-checkpoints":
				opts.v3Checkpoints = next()
					.split(",")
					.map((c) => c.trim())
					.filter(Boolean);
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
				opts.systemFileSupplied = true;   // tracked apart from the value: an explicit
				break;                            // default-path override must still win
			case "--task-file":
				opts.taskFile = resolve(next());
				explicit.add("--task-file");      // tracked before the break: an explicit
				break;                            // default-path override must still win
			case "--help":
			case "-h":
				printHelp();
				process.exit(0);
				break;
			default:
				throw new Error(`unknown option: ${arg}`);
		}
	}
	if (!DATASETS.includes(opts.dataset as any)) {
		throw new Error(`--dataset must be one of ${DATASETS.join("|")}`);
	}
	if (!SEGMENTERS.includes(opts.segmenter as any)) {
		throw new Error(`--segmenter must be ${SEGMENTERS.join("|")} (got ${opts.segmenter})`);
	}
	applyDatasetProfile(opts, explicit);
	return opts;
}

function printHelp(): void {
	console.log(`Usage: npx tsx run_cases.ts [options]

  --preset <name>           load experiments/presets/<name>.json (a logged experiment's recorded model,
                            thinking, cases, budgets, concurrency, dataset, segmenter, v3 flags, prompts);
                            explicit flags below override it (--cases replaces the preset's split and
                            vice versa). --list-presets prints the names.
  --cases F14,M07           comma-separated case ids (or --split)
  --split design|test|all   run every case in that split of the dataset's cases csv
  --dataset <name>          mstt (default, eval/v2/mstt_cases.csv, unchanged behaviour) or beamster
                            (Phase 4: eval/v4/beamster_cases.csv; beamster budget defaults 50 tool
                            calls / 20 prompts / 12 min wall unless overridden by flag)
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
  --segmenter <name>        nninteractive (default) | sammed3d (Phase 6 segmenter swap; the
                            agent-side tools/prompts are identical, only the segmenter changes)
  --segmenter-url <url>     segmenter service URL (default: segd's, http://127.0.0.1:8095 for sammed3d)
  --v3-pointer <mode>       harness v3 pointer for seg_locate: gui_actor|molmo2 (off unless set)
  --v3-pointer-url <url>    (default http://127.0.0.1:8092 gui_actor / :8093 molmo2)
  --v3-medgemma <mode>      harness v3 MedGemma second reader: q4|bf16 (off unless set)
  --v3-medgemma-url <url>   (default http://127.0.0.1:8091 q4 / :8094 bf16)
  --v3-checkpoints <list>   comma list of localize,review,candidates (default: all three;
                            the MedGemma checkpoints injected into tool observations)
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
