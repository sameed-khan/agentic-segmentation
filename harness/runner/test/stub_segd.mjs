#!/usr/bin/env node
/**
 * Tiny stand-in for the segd HTTP API (harness/server/CONTRACTS.md §5).
 *
 * The real Python daemon is written concurrently by another agent. This stub
 * exists so the extension and the runner can be developed and smoke-tested
 * against the HTTP contract before segd is available. It implements only what
 * the agent loop touches:
 *
 *   GET    /healthz
 *   POST   /cases                 -> {ok, handle}
 *   POST   /cases/{h}/op          -> {ok, text, images, done}
 *   GET    /cases/{h}/status      -> {ok, done, tool_calls, model_prompts, ...}
 *   DELETE /cases/{h}
 *
 * Env:
 *   PORT                              listen port (argv[2] wins), 0 = ephemeral
 *   SEGD_STUB_FAIL_FIRST_FINISH=1     make each case's first `finish` fail, so the
 *                                     runner's continuation/nudge path is exercised
 *
 * Run directly:  node test/stub_segd.mjs [port]
 */
import { createServer } from "node:http";

// 24x24 PNG: red circle on a blue background (same idea as checks/ping.png).
const PNG_B64 =
	"iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAIAAABvFaqvAAAAdUlEQVR4nGOUsznBQA3ARBVTqGkQ" +
	"Cx65fW8S0EScRBbgUsyINYwwjSBoHBav4TcFlwJ0gwiagksZE35p4s2iQfST5BxMLYMvQY4Ig/Dk" +
	"I1wAWQttvEaSo9AUo7uISLMwlWHxGkGzsCrAXh5BABUKNjLA4EuQAA/HJbeFXJIHAAAAAElFTkSuQmCC";

const FAIL_FIRST_FINISH = process.env.SEGD_STUB_FAIL_FIRST_FINISH === "1";

/** @type {Map<string, {done:boolean,tool_calls:number,model_prompts:number,current_version:string,finish_failed:boolean}>} */
const cases = new Map();

const MODEL_OPS = new Set(["point", "lasso", "scribble", "box", "refine"]);

function send(res, status, body) {
	res.writeHead(status, { "content-type": "application/json" });
	res.end(JSON.stringify(body));
}

function readBody(req) {
	return new Promise((resolve, reject) => {
		let data = "";
		req.on("data", (chunk) => {
			data += chunk;
			if (data.length > 5_000_000) reject(new Error("body too large"));
		});
		req.on("end", () => {
			if (!data) return resolve({});
			try {
				resolve(JSON.parse(data));
			} catch (error) {
				reject(error);
			}
		});
		req.on("error", reject);
	});
}

function observation(state, op) {
	return (
		`version v1 (parent v0) | volume 12.3 mL | components 1 (largest 100%) | ` +
		`extent axial 40-90, coronal 30-70, sagittal 20-60 | change +12.3 mL / -0.0 mL | ` +
		`prompts used ${state.model_prompts}/15, tool calls ${state.tool_calls}/40\n` +
		`stub ${op}: canned observation; the mask covers the lesion in the current view.`
	);
}

const server = createServer(async (req, res) => {
	const url = new URL(req.url ?? "/", "http://localhost");
	const path = url.pathname;

	try {
		if (req.method === "GET" && path === "/healthz") {
			return send(res, 200, { ok: true });
		}

		if (req.method === "POST" && path === "/cases") {
			const body = await readBody(req);
			const handle = body.case_id ?? body.handle;
			if (!handle) return send(res, 200, { ok: false, error: "case_id required" });
			cases.set(handle, {
				done: false,
				tool_calls: 0,
				model_prompts: 0,
				current_version: "v0",
				finish_failed: false,
			});
			return send(res, 200, { ok: true, handle });
		}

		const opMatch = path.match(/^\/cases\/([^/]+)\/op$/);
		if (req.method === "POST" && opMatch) {
			const handle = decodeURIComponent(opMatch[1]);
			const state = cases.get(handle);
			if (!state) return send(res, 200, { ok: false, error: `unknown case ${handle}` });
			const body = await readBody(req);
			const op = body.op;

			if (op === "open") {
				return send(res, 200, {
					ok: true,
					text:
						"Case card (stub): shape 200x200x120, spacing 1 mm isotropic, sequence T2. " +
						"Display is radiological; marks are labelled. Target: the lesion described in the task.",
					images: [PNG_B64],
					done: false,
				});
			}

			if (op === "finish") {
				if (FAIL_FIRST_FINISH && !state.finish_failed) {
					state.finish_failed = true;
					const error =
						"stub: mask looks small — refine once more before finishing (finish rejected)";
					return send(res, 200, { ok: false, error, text: error });
				}
				state.done = true;
				return send(res, 200, {
					ok: true,
					text: "final mask saved (stub)",
					images: [],
					done: true,
				});
			}

			state.tool_calls += 1;
			if (MODEL_OPS.has(op)) state.model_prompts += 1;
			if (op === "undo" || op === "checkout" || op === "paint" || op === "edit") {
				state.current_version = "v1";
			} else {
				state.current_version = "v1";
			}
			return send(res, 200, {
				ok: true,
				text: observation(state, op),
				images: [PNG_B64],
				done: false,
			});
		}

		const statusMatch = path.match(/^\/cases\/([^/]+)\/status$/);
		if (req.method === "GET" && statusMatch) {
			const handle = decodeURIComponent(statusMatch[1]);
			const state = cases.get(handle);
			if (!state) return send(res, 200, { ok: false, error: `unknown case ${handle}` });
			return send(res, 200, {
				ok: true,
				done: state.done,
				tool_calls: state.tool_calls,
				model_prompts: state.model_prompts,
				current_version: state.current_version,
			});
		}

		const caseMatch = path.match(/^\/cases\/([^/]+)$/);
		if (req.method === "DELETE" && caseMatch) {
			const handle = decodeURIComponent(caseMatch[1]);
			cases.delete(handle);
			return send(res, 200, { ok: true });
		}

		return send(res, 404, { ok: false, error: `no route for ${req.method} ${path}` });
	} catch (error) {
		return send(res, 500, { ok: false, error: String(error) });
	}
});

const port = Number(process.argv[2] ?? process.env.PORT ?? 8791);
server.listen(port, "127.0.0.1", () => {
	const address = server.address();
	const actual = typeof address === "object" && address ? address.port : port;
	console.log(JSON.stringify({ ready: true, url: `http://127.0.0.1:${actual}` }));
});
