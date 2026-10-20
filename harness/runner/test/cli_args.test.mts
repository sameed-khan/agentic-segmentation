/**
 * Regression (review P1 #1 + #2): explicit budget flags (--wall-min, --max-tool-calls,
 * --max-model-prompts) and --task-file used to be recorded AFTER the switch's break, so
 * the set of explicit flags stayed empty and applyDatasetProfile always overrode them —
 * a custom task file (MSTT included) and BEAMSTER budget overrides were silently
 * replaced by the dataset defaults.
 *
 * Run:  cd harness/runner && node --test --import tsx test/cli_args.test.mts
 */
import assert from "node:assert/strict";
import {
	DEFAULT_TASK_FILE,
	DEFAULT_TASK_MULTI_FILE,
	parseArgs,
} from "../cli_args.ts";

const CUSTOM_TASK = "/tmp/ablation/custom_task.md";

// --- defaults (no flags) -----------------------------------------------------
const mstt = parseArgs([]);
assert.equal(mstt.dataset, "mstt");
assert.equal(mstt.wallMin, 10);
assert.equal(mstt.maxToolCalls, 40);
assert.equal(mstt.maxModelPrompts, 15);
assert.equal(mstt.taskFile, DEFAULT_TASK_FILE);

const beam = parseArgs(["--dataset", "beamster"]);
assert.equal(beam.wallMin, 12);
assert.equal(beam.maxToolCalls, 50);
assert.equal(beam.maxModelPrompts, 20);
assert.equal(beam.taskFile, DEFAULT_TASK_MULTI_FILE);

// --- explicit --task-file wins on both datasets (incl. the default path) ------
assert.equal(parseArgs(["--task-file", CUSTOM_TASK]).taskFile, CUSTOM_TASK);
assert.equal(parseArgs(["--dataset", "beamster", "--task-file", CUSTOM_TASK]).taskFile, CUSTOM_TASK);
// explicitly passing the dataset's default path is still an explicit choice
assert.equal(parseArgs(["--dataset", "mstt", "--task-file", DEFAULT_TASK_FILE]).taskFile, DEFAULT_TASK_FILE);
assert.equal(parseArgs(["--dataset", "beamster", "--task-file", DEFAULT_TASK_MULTI_FILE]).taskFile,
	DEFAULT_TASK_MULTI_FILE);

// --- each budget override alone beats the dataset default ---------------------
for (const [flag, key, value] of [
	["--wall-min", "wallMin", "7"],
	["--max-tool-calls", "maxToolCalls", "8"],
	["--max-model-prompts", "maxModelPrompts", "3"],
] as const) {
	const o = parseArgs(["--dataset", "beamster", flag, value]);
	assert.equal(o[key], Number(value), `${flag} alone`);
	// ...and the other two still take the beamster defaults
	const others = { wallMin: 12, maxToolCalls: 50, maxModelPrompts: 20 } as Record<string, number>;
	delete others[key];
	for (const [k, v] of Object.entries(others)) assert.equal(o[k as keyof typeof o], v, `${flag} alone: ${k} default`);
}

// --- combined overrides, both flag orders relative to --dataset ----------------
for (const argv of [
	["--dataset", "beamster", "--wall-min", "2", "--max-tool-calls", "8", "--max-model-prompts", "3"],
	["--wall-min", "2", "--max-tool-calls", "8", "--max-model-prompts", "3", "--dataset", "beamster"],
	["--wall-min", "2", "--dataset", "beamster", "--max-tool-calls", "8", "--max-model-prompts", "3"],
]) {
	const o = parseArgs(argv);
	assert.equal(o.wallMin, 2);
	assert.equal(o.maxToolCalls, 8);
	assert.equal(o.maxModelPrompts, 3);
}
// task-file + budgets combined, after --dataset
const all = parseArgs(["--dataset", "beamster", "--task-file", CUSTOM_TASK,
	"--wall-min", "2", "--max-tool-calls", "8", "--max-model-prompts", "3"]);
assert.equal(all.taskFile, CUSTOM_TASK);
assert.equal(all.wallMin, 2);
assert.equal(all.maxToolCalls, 8);
assert.equal(all.maxModelPrompts, 3);

console.log("ok: explicit budget/task-file flags beat the dataset defaults in any order");
