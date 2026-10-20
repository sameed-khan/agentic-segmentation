/**
 * Harness v3 tool-declaration test: `seg_locate` is registered only when the case's
 * v3 flags enable a pointer; with v3 off the declaration is byte-identical to v2.
 *
 * Run:  cd harness/extension && ../runner/node_modules/.bin/tsx test_v3_tools.ts
 * (exit 0 + "ok" on success; assertion error otherwise)
 */
import assert from "node:assert/strict";
import { createSegExtension, SEG_TOOL_NAMES } from "./index.ts";

function registeredTools(options: Parameters<typeof createSegExtension>[0]): string[] {
	const names: string[] = [];
	const pi = {
		registerTool: (tool: { name: string }) => names.push(tool.name),
		on: () => undefined,
	} as unknown as ExtensionLike;
	createSegExtension(options)(pi as never);
	return names;
}

interface ExtensionLike {
	registerTool: (tool: { name: string }) => void;
	on: (event: string, handler: (payload: unknown) => void) => void;
}

const base = [...SEG_TOOL_NAMES];
assert.equal(base.includes("seg_locate"), false, "v2 tool list must not contain seg_locate");

// v3 off: identical declaration order
const off = registeredTools({ segdUrl: "http://127.0.0.1:1", caseHandle: "c" });
assert.deepEqual(off, base);

// all-off v3 dict: still identical
const off2 = registeredTools({ caseHandle: "c", v3: { pointer: null } });
assert.deepEqual(off2, base);

// pointer on: seg_locate inserted right after seg_view, everything else unchanged
for (const pointer of ["gui_actor", "molmo2"]) {
	const on = registeredTools({ caseHandle: "c", v3: { pointer } });
	assert.deepEqual(on, [...base.slice(0, 3), "seg_locate", ...base.slice(3)]);
}

console.log(`ok: seg_locate declared only when a v3 pointer is set (${base.length} tools v2, +1 v3)`);
