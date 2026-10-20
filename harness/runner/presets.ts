/**
 * Experiment presets for run_cases.ts (`--preset <name>`).
 *
 * A preset is `experiments/presets/<name>.json` (kind "agent_run"): the complete, recorded
 * configuration of one logged experiment — model, thinking level, case selection, budgets,
 * concurrency, dataset, segmenter, v3 flags, system/task prompt files — plus provenance (the
 * git tag/commit the original run used, results artifact, experiment-log entry). It is expanded
 * into ordinary CLI flags that go BEFORE the user's own flags, so every explicit flag overrides
 * the preset (and `--cases` replaces the preset's `split`, `--split` its `cases`).
 *
 * Operational settings (segd/nnInteractive/segmenter service URLs, results dir, auth shim) are
 * deliberately not part of a preset: they describe the machine, not the experiment.
 *
 * Own module (like cli_args.ts) so loading/expansion is unit-testable without run_cases.ts.
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const PROJECT_ROOT = resolve(HERE, "..", "..");
export const PRESET_DIR = join(PROJECT_ROOT, "experiments", "presets");

export interface PresetV3 {
	pointer?: string | null;
	pointer_url?: string | null;
	medgemma?: string | null;
	medgemma_url?: string | null;
	checkpoints?: string[];
}

export interface PresetRunner {
	run_id: string;
	model: string;
	thinking: string;
	dataset: string;
	split?: string;
	cases?: string[];
	concurrency: number;
	wall_min: number;
	max_nudges: number;
	max_tool_calls: number;
	max_model_prompts: number;
	segmenter: string;
	segmenter_url?: string | null;
	v3: PresetV3 | null;
	system_file: string;
	task_file: string;
	system_prompt_sha256?: string;
	note: string;
}

export interface PresetProvenance {
	git_tag: string;
	git_commit: string;
	git_describe?: string;
	results: string;
	log_entry: string;
	[key: string]: unknown;
}

export interface AgentRunPreset {
	schema: number;
	name: string;
	kind: "agent_run";
	description: string;
	provenance: PresetProvenance;
	runner: PresetRunner;
}

const RUNNER_REQUIRED = [
	"run_id", "model", "thinking", "dataset", "concurrency", "wall_min", "max_nudges",
	"max_tool_calls", "max_model_prompts", "segmenter", "v3", "system_file", "task_file", "note",
] as const;
const RUNNER_OPTIONAL = ["split", "cases", "segmenter_url", "system_prompt_sha256"] as const;
const V3_KEYS = ["pointer", "pointer_url", "medgemma", "medgemma_url", "checkpoints"];
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** All preset names (file stems) in the directory, sorted. */
export function listPresetNames(dir: string = PRESET_DIR): string[] {
	if (!existsSync(dir)) return [];
	return readdirSync(dir).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5)).sort();
}

/** Validate a parsed preset object (throws a clear Error naming the preset). */
export function validatePreset(p: any, name: string): AgentRunPreset {
	const bad = (msg: string): never => {
		throw new Error(`preset ${name}: ${msg}`);
	};
	if (!p || typeof p !== "object") bad("not a JSON object");
	if (p.schema !== 1) bad(`schema must be 1 (got ${p.schema})`);
	if (p.name !== name) bad(`"name" is ${JSON.stringify(p.name)}, file name says ${JSON.stringify(name)}`);
	if (p.kind !== "agent_run") bad(`kind ${JSON.stringify(p.kind)} is not runnable by run_cases.ts (only "agent_run")`);
	for (const k of ["description", "provenance", "runner"]) if (p[k] === undefined) bad(`missing "${k}"`);
	for (const k of ["git_tag", "git_commit", "results", "log_entry"]) {
		if (typeof p.provenance[k] !== "string" || !p.provenance[k]) bad(`provenance.${k} missing`);
	}
	const r = p.runner;
	for (const k of RUNNER_REQUIRED) if (!(k in r)) bad(`runner.${k} missing`);
	for (const k of Object.keys(r)) {
		if (!(RUNNER_REQUIRED as readonly string[]).includes(k) && !(RUNNER_OPTIONAL as readonly string[]).includes(k)) {
			bad(`unknown runner key ${JSON.stringify(k)}`);
		}
	}
	if ((r.split === undefined) === (r.cases === undefined)) bad('exactly one of runner.split / runner.cases');
	if (r.cases !== undefined && !(Array.isArray(r.cases) && r.cases.length > 0 && r.cases.every((c: any) => typeof c === "string"))) {
		bad("runner.cases must be a non-empty string array");
	}
	if (r.v3 !== null) {
		if (typeof r.v3 !== "object") bad("runner.v3 must be null or an object");
		for (const k of Object.keys(r.v3)) if (!V3_KEYS.includes(k)) bad(`unknown runner.v3 key ${JSON.stringify(k)}`);
	}
	return p as AgentRunPreset;
}

export function loadPreset(name: string, dir: string = PRESET_DIR): AgentRunPreset {
	if (!NAME_RE.test(name)) throw new Error(`invalid preset name ${JSON.stringify(name)}`);
	const file = join(dir, `${name}.json`);
	if (!existsSync(file)) {
		throw new Error(`unknown preset ${JSON.stringify(name)} (${file} not found; --list-presets shows the available ones)`);
	}
	return validatePreset(JSON.parse(readFileSync(file, "utf8")), name);
}

const absolute = (path: string): string => (isAbsolute(path) ? path : join(PROJECT_ROOT, path));

/**
 * The CLI flags a preset stands for. `userFlags` = the flags the user passed explicitly:
 * the case selector the user did not choose is dropped, so `--cases A,B` (or `--split x`)
 * replaces the preset's selection instead of conflicting with it.
 */
export function presetToArgv(preset: AgentRunPreset, userFlags: ReadonlySet<string> = new Set()): string[] {
	const r = preset.runner;
	const a: string[] = [
		"--run-id", r.run_id,
		"--model", r.model,
		"--thinking", r.thinking,
		"--dataset", r.dataset,
		"--concurrency", String(r.concurrency),
		"--wall-min", String(r.wall_min),
		"--max-nudges", String(r.max_nudges),
		"--max-tool-calls", String(r.max_tool_calls),
		"--max-model-prompts", String(r.max_model_prompts),
		"--segmenter", r.segmenter,
		"--system-file", absolute(r.system_file),
		"--task-file", absolute(r.task_file),
		"--note", r.note,
	];
	if (r.segmenter_url) a.push("--segmenter-url", r.segmenter_url);
	if (r.cases !== undefined && !userFlags.has("--split")) a.push("--cases", r.cases.join(","));
	if (r.split !== undefined && !userFlags.has("--cases")) a.push("--split", r.split);
	const v3 = r.v3;
	if (v3) {
		if (v3.pointer) a.push("--v3-pointer", v3.pointer);
		if (v3.pointer_url) a.push("--v3-pointer-url", v3.pointer_url);
		if (v3.medgemma) a.push("--v3-medgemma", v3.medgemma);
		if (v3.medgemma_url) a.push("--v3-medgemma-url", v3.medgemma_url);
		if (v3.checkpoints && v3.checkpoints.length) a.push("--v3-checkpoints", v3.checkpoints.join(","));
	}
	return a;
}
