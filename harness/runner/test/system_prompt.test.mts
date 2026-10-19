/**
 * Regression (review P2 #10): an explicit `--system-file` wins even when its resolved
 * path equals the default — passing harness/prompts/system.md with v3 enabled used to
 * select system_v3.md because selection only compared paths.
 *
 * Run:  cd harness/runner && npx tsx test/system_prompt.test.mts
 * (exit 0 + "ok" on success; assertion error otherwise)
 */
import assert from "node:assert/strict";
import {
	DEFAULT_SYSTEM_FILE,
	DEFAULT_SYSTEM_V3_FILE,
	selectSystemFile,
} from "../system_prompt.ts";

const baseline = "/tmp/ablation/baseline.md";

// no explicit flag: default for v2, v3 variant when v3 is on
assert.equal(selectSystemFile(false, DEFAULT_SYSTEM_FILE, false), DEFAULT_SYSTEM_FILE);
assert.equal(selectSystemFile(false, DEFAULT_SYSTEM_FILE, true), DEFAULT_SYSTEM_V3_FILE);
// explicit flag always wins — including when it EQUALS the default path (the bug)
assert.equal(selectSystemFile(true, baseline, true), baseline);
assert.equal(selectSystemFile(true, baseline, false), baseline);
assert.equal(selectSystemFile(true, DEFAULT_SYSTEM_FILE, true), DEFAULT_SYSTEM_FILE);

console.log("ok: explicit --system-file wins over the v3 default selection");
