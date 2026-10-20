/**
 * Preset loader/expansion (--preset <name>): every experiments/presets/*.json agent_run preset
 * resolves through parseArgs to exactly its recorded settings; explicit flags override;
 * malformed presets fail with a clear message.
 *
 * Run:  cd harness/runner && node --test --import tsx test/presets.test.mts
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { DATASETS, parseArgs } from "../cli_args.ts";
import {
	listPresetNames,
	loadPreset,
	PRESET_DIR,
	PROJECT_ROOT,
	presetToArgv,
	validatePreset,
	type AgentRunPreset,
} from "../presets.ts";

const names = listPresetNames();
const csvRows = (file: string): { case: string; split: string }[] => {
	const [head, ...lines] = readFileSync(join(PROJECT_ROOT, file), "utf8").trim().split(/\r?\n/);
	const cols = head.split(",");
	return lines.map((l) => {
		const cells = l.split(",");
		return { case: cells[cols.indexOf("case")], split: (cells[cols.indexOf("split")] ?? "").trim() };
	});
};
const abs = (p: string): string => join(PROJECT_ROOT, p);
const renderSha = (file: string, tool: number, prompts: number): string =>
	createHash("sha256")
		.update(
			readFileSync(file, "utf8").replace(/\{\{(\w+)\}\}/g, (m, k: string) =>
				k === "max_tool_calls" ? String(tool) : k === "max_model_prompts" ? String(prompts) : m,
			),
		)
		.digest("hex");

test("the experiment presets exist (phase3 + sweep + p6)", () => {
	for (const n of ["phase3-high", "phase3-xhigh", "p3-opus55-medium", "p3-gpt6luna-max", "p6-sammed3d-high"]) {
		assert.ok(names.includes(n), `preset ${n} missing`);
	}
	assert.ok(names.length >= 18);
});

for (const name of names) {
	test(`preset ${name} resolves to its recorded settings`, () => {
		const p = loadPreset(name);
		const r = p.runner;
		const o = parseArgs(["--preset", name]);
		assert.equal(o.preset, name);
		assert.equal(o.runId, r.run_id);
		assert.equal(o.model, r.model);
		assert.equal(o.thinking, r.thinking);
		assert.equal(o.dataset, r.dataset);
		assert.ok((DATASETS as readonly string[]).includes(o.dataset));
		assert.equal(o.concurrency, r.concurrency);
		assert.equal(o.wallMin, r.wall_min);
		assert.equal(o.maxNudges, r.max_nudges);
		assert.equal(o.maxToolCalls, r.max_tool_calls);
		assert.equal(o.maxModelPrompts, r.max_model_prompts);
		assert.equal(o.segmenter, r.segmenter);
		assert.equal(o.segmenterUrl, r.segmenter_url ?? null);
		assert.equal(o.note, r.note);
		assert.equal(o.split, r.split ?? null);
		assert.deepEqual(o.cases, r.cases ?? []);
		assert.equal(o.systemFile, abs(r.system_file));
		assert.equal(o.taskFile, abs(r.task_file));
		assert.equal(o.systemFileSupplied, true);
		assert.ok(existsSync(o.systemFile) && existsSync(o.taskFile));
		// v3 flags: recorded null = off
		if (r.v3 === null) {
			assert.equal(o.v3Pointer, null);
			assert.equal(o.v3Medgemma, null);
		}
		// the recorded system prompt hash is the rendered current prompt file (budget-substituted)
		if (r.system_prompt_sha256) {
			assert.equal(renderSha(o.systemFile, r.max_tool_calls, r.max_model_prompts), r.system_prompt_sha256);
		}
		// case selection is valid against the dataset's manifest
		const csv = r.dataset === "mstt" ? "eval/v2/mstt_cases.csv" : `eval/v4/${r.dataset}_cases.csv`;
		const rows = csvRows(csv);
		if (r.cases) for (const c of r.cases) assert.ok(rows.some((x) => x.case === c), `${c} not in ${csv}`);
		else assert.ok(rows.some((x) => x.split === r.split) || r.split === "all");
	});
}

test("p3-* / phase3-* / p6 settings pinned (model, thinking, concurrency, segmenter)", () => {
	const want: Record<string, [string, string, number, string]> = {
		"phase3-high": ["anthropic/claude-sonnet-5-5", "high", 4, "nninteractive"],
		"phase3-xhigh": ["anthropic/claude-sonnet-5-5", "xhigh", 4, "nninteractive"],
		"p3-opus55-medium": ["anthropic/claude-opus-5-5", "medium", 8, "nninteractive"],
		"p3-opus55-high": ["anthropic/claude-opus-5-5", "high", 8, "nninteractive"],
		"p3-opus55-xhigh": ["anthropic/claude-opus-5-5", "xhigh", 8, "nninteractive"],
		"p3-haiku55-xhigh": ["anthropic/claude-haiku-5-5", "xhigh", 8, "nninteractive"],
		"p3-haiku55-max": ["anthropic/claude-haiku-5-5", "max", 8, "nninteractive"],
		"p3-dsv41flash-xhigh": ["opencode-go/deepseek-v4.1-flash", "xhigh", 8, "nninteractive"],
		"p3-dsv41flash-max": ["opencode-go/deepseek-v4.1-flash", "max", 8, "nninteractive"],
		"p3-muse13-high": ["opencode-go/muse-spark-1.3-contributor", "high", 8, "nninteractive"],
		"p3-muse13-xhigh": ["opencode-go/muse-spark-1.3-contributor", "xhigh", 8, "nninteractive"],
		"p3-gpt61sol-high": ["openai/gpt-6.1-sol", "high", 8, "nninteractive"],
		"p3-gpt61sol-xhigh": ["openai/gpt-6.1-sol", "xhigh", 8, "nninteractive"],
		"p3-gpt6luna-xhigh": ["openai/gpt-6-luna", "xhigh", 8, "nninteractive"],
		"p3-gpt6luna-max": ["openai/gpt-6-luna", "max", 8, "nninteractive"],
		"p6-sammed3d-high": ["anthropic/claude-sonnet-5-5", "high", 6, "sammed3d"],
	};
	for (const [n, [model, thinking, conc, seg]] of Object.entries(want)) {
		const o = parseArgs(["--preset", n]);
		assert.deepEqual([o.model, o.thinking, o.concurrency, o.segmenter], [model, thinking, conc, seg], n);
		assert.equal(o.split, n === "p3-gpt61sol-xhigh" ? null : "test", `${n} split`);
		assert.equal(o.v3, null);
	}
	// xhigh probe: the 30 recorded probe cases (6 per histotype)
	const probe = parseArgs(["--preset", "p3-gpt61sol-xhigh"]);
	assert.equal(probe.cases.length, 30);
	assert.equal(probe.cases[0], "F13");
	assert.equal(probe.segmenterUrl, null);
	assert.equal(parseArgs(["--preset", "p6-sammed3d-high"]).segmenterUrl, "http://127.0.0.1:8095");
});

test("explicit flags override the preset; --cases replaces split and vice versa", () => {
	const base = parseArgs(["--preset", "p3-opus55-high"]);
	assert.equal(base.split, "test");
	const o = parseArgs(["--preset", "p3-opus55-high", "--model", "x/y", "--concurrency", "1", "--run-id", "mine",
		"--thinking", "low", "--max-tool-calls", "5"]);
	assert.deepEqual([o.model, o.concurrency, o.runId, o.thinking, o.maxToolCalls], ["x/y", 1, "mine", "low", 5]);
	assert.equal(o.maxModelPrompts, 15);            // untouched preset value
	const c = parseArgs(["--preset", "p3-opus55-high", "--cases", "F02,F06"]);
	assert.deepEqual(c.cases, ["F02", "F06"]);
	assert.equal(c.split, null);                    // preset's split dropped
	const s = parseArgs(["--preset", "p6-sammed3d-pilot5b", "--split", "design"]);
	assert.equal(s.split, "design");
	assert.deepEqual(s.cases, []);                  // preset's cases dropped
	// flag order relative to --preset does not matter
	assert.equal(parseArgs(["--model", "x/y", "--preset", "p3-opus55-high"]).model, "x/y");
	// segmenter override composes with the other recorded settings
	const seg = parseArgs(["--preset", "p3-opus55-high", "--segmenter", "sammed3d"]);
	assert.equal(seg.segmenter, "sammed3d");
	assert.equal(seg.model, "anthropic/claude-opus-5-5");
});

test("no preset = harness-v2.0.1 defaults (v3 off, dataset mstt, segmenter nninteractive)", () => {
	const o = parseArgs([]);
	assert.equal(o.preset, null);
	assert.deepEqual([o.dataset, o.segmenter, o.v3Pointer, o.v3Medgemma, o.maxToolCalls, o.maxModelPrompts, o.wallMin],
		["mstt", "nninteractive", null, null, 40, 15, 10]);
	assert.throws(() => parseArgs(["--segmenter", "medsam2"]), /--segmenter must be/);
});

function synthetic(over: Record<string, unknown> = {}, runner: Record<string, unknown> = {}): AgentRunPreset {
	return {
		schema: 1, name: "synthetic", kind: "agent_run", description: "d",
		provenance: { git_tag: "t", git_commit: "c", results: "r", log_entry: "E0" },
		runner: {
			run_id: "synthetic", model: "a/b", thinking: "high", dataset: "mstt", split: "design", concurrency: 2,
			wall_min: 10, max_nudges: 3, max_tool_calls: 40, max_model_prompts: 15, segmenter: "nninteractive", v3: null,
			system_file: "harness/prompts/system.md", task_file: "harness/prompts/task.md", note: "n", ...runner,
		},
		...over,
	} as AgentRunPreset;
}

test("v3, beamster and sammed3d settings expand into the matching flags", () => {
	const v3 = synthetic({}, {
		system_file: "harness/prompts/system_v3.md",
		v3: { pointer: "molmo2", medgemma: "bf16", checkpoints: ["review", "localize"] },
		segmenter: "sammed3d", segmenter_url: "http://127.0.0.1:9999",
	});
	const argv = presetToArgv(validatePreset(v3, "synthetic"));
	const o = parseArgs(argv);
	assert.deepEqual([o.v3Pointer, o.v3Medgemma, o.v3Checkpoints], ["molmo2", "bf16", ["review", "localize"]]);
	assert.deepEqual([o.segmenter, o.segmenterUrl], ["sammed3d", "http://127.0.0.1:9999"]);
	const beam = synthetic({}, { dataset: "beamster", system_file: "harness/prompts/system_multi.md",
		task_file: "harness/prompts/task_multi.md", max_tool_calls: 50, max_model_prompts: 20, wall_min: 12 });
	const ob = parseArgs(presetToArgv(validatePreset(beam, "synthetic")));
	assert.deepEqual([ob.dataset, ob.maxToolCalls, ob.maxModelPrompts, ob.wallMin], ["beamster", 50, 20, 12]);
});

test("malformed presets and unknown names fail clearly", () => {
	assert.throws(() => loadPreset("does-not-exist"), /unknown preset "does-not-exist"/);
	assert.throws(() => loadPreset("../x"), /invalid preset name/);
	assert.throws(() => parseArgs(["--preset"]), /missing value for --preset/);
	assert.throws(() => parseArgs(["--preset", "a", "--preset", "b"]), /given twice/);
	assert.throws(() => validatePreset(synthetic({ kind: "study" }), "synthetic"), /not runnable by run_cases/);
	assert.throws(() => validatePreset(synthetic({ name: "other" }), "synthetic"), /file name says/);
	assert.throws(() => validatePreset(synthetic({}, { cases: ["F02"] }), "synthetic"), /exactly one of/);
	assert.throws(() => validatePreset(synthetic({}, { typo_key: 1 }), "synthetic"), /unknown runner key/);
	const bad = synthetic();
	delete (bad.runner as any).concurrency;
	assert.throws(() => validatePreset(bad, "synthetic"), /runner.concurrency missing/);
	assert.throws(() => validatePreset(synthetic({}, { v3: { bogus: 1 } }), "synthetic"), /unknown runner.v3 key/);
	// a study preset on disk is not listed/loadable by the agent runner
	const dir = mkdtempSync(join(tmpdir(), "presets-"));
	writeFileSync(join(dir, "studyx.json"), JSON.stringify({ schema: 1, name: "studyx", kind: "study" }));
	assert.throws(() => loadPreset("studyx", dir), /missing "provenance"|not runnable/);
	assert.ok(PRESET_DIR.endsWith("experiments/presets"));
});
