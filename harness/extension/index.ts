/**
 * pi extension for the segmentation harness.
 *
 * Registers one `seg_*` tool per segd operation. Each tool is a thin HTTP call
 * to `${SEGD_URL}/cases/${SEGD_CASE}/op` and returns the daemon's text plus any
 * PNG images inline, so the model never needs a separate read turn.
 *
 * The case handle is bound per factory call, so one Node process can run
 * several cases concurrently with independent handles:
 *
 *   const factory = createSegExtension({ segdUrl, caseHandle });
 *   pi.registerExtension? -> factory(pi)
 *
 * The default export derives the handle from `SEGD_URL` / `SEGD_CASE` for the
 * `harness/bin/pi-seg` CLI wrapper.
 */
import { Type } from "@earendil-works/pi-ai";
import {
	defineTool,
	type ExtensionAPI,
	type ExtensionFactory,
	type ToolDefinition,
} from "@earendil-works/pi-coding-agent";

/** Default segd address; matches CONTRACTS `harness/server/CONTRACTS.md` §5. */
export const SEGD_DEFAULT_URL = "http://127.0.0.1:8765";

/** Per-call timeout for a segd operation. Rendering and nnInteractive can be slow. */
export const SEGD_OP_TIMEOUT_MS = 180_000;

/** The tool names this extension registers, in the order they are declared. */
export const SEG_TOOL_NAMES = [
	"seg_open",
	"seg_sheet",
	"seg_view",
	"seg_point",
	"seg_lasso",
	"seg_scribble",
	"seg_box",
	"seg_paint",
	"seg_refine",
	"seg_edit",
	"seg_undo",
	"seg_checkout",
	"seg_finish",
] as const;

export interface SegExtensionOptions {
	/** segd base URL. Defaults to `SEGD_DEFAULT_URL`. */
	segdUrl?: string;
	/** Case handle returned by `POST /cases`. From `SEGD_CASE` in the CLI wrapper. */
	caseHandle?: string;
	/**
	 * Optional observer for the raw provider request payload. The runner uses it
	 * to record the system prompt that reaches the provider. Never throws into
	 * the agent loop.
	 */
	onProviderRequest?: (payload: unknown) => void;
}

/** One inline image block understood by pi tool results. */
interface ImageBlock {
	type: "image";
	data: string;
	mimeType: string;
}

interface TextBlock {
	type: "text";
	text: string;
}

interface ToolResult {
	content: Array<TextBlock | ImageBlock>;
	details: Record<string, unknown>;
}

interface OpResponse {
	ok?: boolean;
	text?: string;
	error?: string;
	images?: string[];
	done?: boolean;
	handle?: string;
}

const AXIS = Type.Union([
	Type.Literal("axial"),
	Type.Literal("coronal"),
	Type.Literal("sagittal"),
]);
const POLARITY = Type.Union([Type.Literal("positive"), Type.Literal("negative")]);
const CROP = Type.Union([
	Type.Literal("full"),
	Type.Literal("mask"),
	Type.Literal("marks"),
]);
const OVERLAY = Type.Union([
	Type.Literal("none"),
	Type.Literal("outline"),
	Type.Literal("fill"),
]);
const MARKS = Type.Union([
	Type.Literal("none"),
	Type.Literal("coarse"),
	Type.Literal("fine"),
	Type.Literal("edges"),
]);
const PAINT_MODE = Type.Union([Type.Literal("add"), Type.Literal("erase")]);
const EDIT_OPS = Type.Union([
	Type.Literal("keep_component"),
	Type.Literal("remove_component"),
	Type.Literal("fill_holes"),
	Type.Literal("close"),
	Type.Literal("open"),
	Type.Literal("dilate"),
	Type.Literal("erode"),
]);
const CONFIDENCE = Type.Union([
	Type.Literal("high"),
	Type.Literal("medium"),
	Type.Literal("low"),
]);

/** A mark id: integer for a global mark, `"E<n>"` for an edge mark. */
const MARK_ID = Type.Union([Type.Integer(), Type.String()]);
const MARK_IDS = Type.Array(MARK_ID, {
	description: "Mark ids shown in the current seg_view, in order.",
});
/** Slice index, or the slice through the current mask's largest cross-section. */
const SLICE = Type.Union([Type.Integer(), Type.Literal("mask")]);

function errorResult(op: string, message: string): ToolResult {
	return {
		content: [{ type: "text", text: `seg ${op} failed: ${message}` }],
		details: { op, ok: false, error: message },
	};
}

function normalizeBaseUrl(url: string): string {
	return url.replace(/\/+$/, "");
}

export function createSegExtension(options: SegExtensionOptions): ExtensionFactory {
	const baseUrl = normalizeBaseUrl(options.segdUrl ?? SEGD_DEFAULT_URL);
	const caseHandle = options.caseHandle;

	async function runOp(
		op: string,
		args: Record<string, unknown>,
	): Promise<ToolResult> {
		if (!caseHandle) {
			return errorResult(op, "no case handle configured (set SEGD_CASE)");
		}
		const url = `${baseUrl}/cases/${encodeURIComponent(caseHandle)}/op`;
		let response: Response;
		try {
			response = await fetch(url, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({ op, args }),
				signal: AbortSignal.timeout(SEGD_OP_TIMEOUT_MS),
			});
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			return errorResult(op, `segd request error: ${message}`);
		}

		const raw = await response.text();
		let body: OpResponse;
		try {
			body = JSON.parse(raw) as OpResponse;
		} catch {
			return errorResult(
				op,
				`segd returned non-JSON (HTTP ${response.status}): ${raw.slice(0, 400)}`,
			);
		}

		if (body.ok === false) {
			return errorResult(op, body.error ?? body.text ?? "unknown segd error");
		}

		const content: Array<TextBlock | ImageBlock> = [
			{ type: "text", text: body.text ?? "(no text)" },
		];
		for (const data of body.images ?? []) {
			content.push({ type: "image", data, mimeType: "image/png" });
		}
		if (body.done) {
			content.push({
				type: "text",
				text:
					op === "finish"
						? "Case finished. The final mask has been saved; do not call any more tools."
						: "Case finished.",
			});
		}
		return {
			content,
			details: { op, ok: true, done: body.done === true },
		};
	}

	function asArgs(params: unknown): Record<string, unknown> {
		return params as Record<string, unknown>;
	}

	const tools: Array<ToolDefinition<any, any, any>> = [
		defineTool({
			name: "seg_open",
			label: "Open case",
			description:
				"Read the case card (volume shape, spacing, sequence, target text, mark convention, budgets) and get an orthogonal overview with orientation letters and a 10 mm scale bar. Call this first.",
			parameters: Type.Object({}),
			async execute(_id, _params) {
				return runOp("open", {});
			},
		}),
		defineTool({
			name: "seg_sheet",
			label: "Slice sheet",
			description:
				"Contact sheet of evenly spaced slices along one axis, each labelled with its slice index. Use it to locate the lesion, then inspect single slices with seg_view.",
			parameters: Type.Object({
				axis: AXIS,
				start: Type.Integer({ description: "First slice index (inclusive)." }),
				stop: Type.Integer({ description: "Last slice index (inclusive)." }),
				n: Type.Integer({ description: "Number of slices to show." }),
			}),
			async execute(_id, params) {
				return runOp("sheet", asArgs(params));
			},
		}),
		defineTool({
			name: "seg_view",
			label: "View slice",
			description:
				"Show one slice as a clean panel and a marked panel. Clean + marked are returned side by side. All mark ids refer to the view returned last. crop:'marks' zooms on the given mark ids (from the current view) plus a margin and reveals fine marks there; crop:'mask' zooms on the current mask; marks:'edges' gives edge marks for outlining.",
			parameters: Type.Object({
				axis: AXIS,
				slice: SLICE,
				crop: CROP,
				overlay: OVERLAY,
				marks: MARKS,
				ids: Type.Optional(
					Type.Array(MARK_ID, {
						description:
							"Required with crop:'marks': mark ids (int or \"E<n>\") shown in the current view to zoom on.",
					}),
				),
			}),
			async execute(_id, params) {
				return runOp("view", asArgs(params));
			},
		}),
		defineTool({
			name: "seg_point",
			label: "Point prompt",
			description:
				"Run the segmentation model from point prompt(s) on the current view. Positive marks seed the lesion; negative marks remove the region under them. Adds to the current mask.",
			parameters: Type.Object({
				ids: MARK_IDS,
				polarity: POLARITY,
			}),
			async execute(_id, params) {
				return runOp("point", asArgs(params));
			},
		}),
		defineTool({
			name: "seg_lasso",
			label: "Lasso prompt",
			description:
				"Run the model from a lasso: at least 3 ordered mark ids on the current slice form a closed polygon around (positive) or inside (negative) the lesion. Prefer this over many points for a rough outline.",
			parameters: Type.Object({
				ids: Type.Array(MARK_ID, {
					description: "At least 3 mark ids on the current slice, listed in order around the outline (clockwise or counter-clockwise); the polygon connects them in this order and closes back to the first.",
					minItems: 3,
				}),
				polarity: POLARITY,
			}),
			async execute(_id, params) {
				return runOp("lasso", asArgs(params));
			},
		}),
		defineTool({
			name: "seg_scribble",
			label: "Scribble prompt",
			description:
				"Run the model from a scribble: ordered mark ids on the current slice form a polyline. Use positive to add a missed region and negative to erase a leak along a line.",
			parameters: Type.Object({
				ids: Type.Array(MARK_ID, { description: "At least 2 mark ids on the current slice, listed in the order the line should run; the polyline connects them in this order.", minItems: 2 }),
				polarity: POLARITY,
			}),
			async execute(_id, params) {
				return runOp("scribble", asArgs(params));
			},
		}),
		defineTool({
			name: "seg_box",
			label: "Box prompt",
			description:
				"Run the model from a 2D box. Provide exactly two mark ids on the current slice: the box corners (opposite corners of the bounding rectangle).",
			parameters: Type.Object({
				ids: Type.Array(MARK_ID, {
					description: "Exactly two mark ids: the box corners.",
					minItems: 2,
					maxItems: 2,
				}),
			}),
			async execute(_id, params) {
				const ids = (params as { ids: Array<number | string> }).ids;
				// CONTRACTS §6 maps this op to box(id_a, id_b).
				return runOp("box", { id_a: ids[0], id_b: ids[1] });
			},
		}),
		defineTool({
			name: "seg_paint",
			label: "Paint mask",
			description:
				"Edit the mask directly without the model: rasterize the polygon through ordered mark ids on the current slice and apply it to the slice range. mode add fills, mode erase removes. Use seg_refine afterwards to let the model clean the painted boundary.",
			parameters: Type.Object({
				ids: Type.Array(MARK_ID, { description: "At least 3 mark ids on the current slice, listed in order around the outline (clockwise or counter-clockwise); the polygon connects them in this order and closes back to the first.", minItems: 3 }),
				mode: PAINT_MODE,
				slice_range: Type.Array(Type.Integer(), {
					description: "Two slice indices [start, stop] on the current axis, both inclusive.",
					minItems: 2,
					maxItems: 2,
				}),
			}),
			async execute(_id, params) {
				return runOp("paint", asArgs(params));
			},
		}),
		defineTool({
			name: "seg_refine",
			label: "Refine mask",
			description:
				"Send the current mask to the model as an initial segmentation so it refines the boundary. Use after seg_paint or a rough lasso.",
			parameters: Type.Object({}),
			async execute(_id, _params) {
				return runOp("refine", {});
			},
		}),
		defineTool({
			name: "seg_edit",
			label: "Edit mask",
			description:
				"Non-model mask edit. keep_component/remove_component need a mark id inside the component. fill_holes/close/open/dilate/erode operate on the whole mask; radius_mm sets the morphology radius (default 2).",
			parameters: Type.Object({
				op: EDIT_OPS,
				mark: Type.Optional(MARK_ID),
				radius_mm: Type.Optional(Type.Number()),
			}),
			async execute(_id, params) {
				return runOp("edit", asArgs(params));
			},
		}),
		defineTool({
			name: "seg_undo",
			label: "Undo",
			description: "Restore the parent of the current mask version.",
			parameters: Type.Object({}),
			async execute(_id, _params) {
				return runOp("undo", {});
			},
		}),
		defineTool({
			name: "seg_checkout",
			label: "Checkout version",
			description:
				"Restore a specific mask version id reported in earlier observations (e.g. v3).",
			parameters: Type.Object({
				version: Type.String({ description: "Version id, e.g. v3." }),
			}),
			async execute(_id, params) {
				return runOp("checkout", asArgs(params));
			},
		}),
		defineTool({
			name: "seg_finish",
			label: "Finish case",
			description:
				"Save the final mask and end the case. The first call (without review_code) shows a final review of the current mask (slices across each plane and just past the mask ends) and a review code. Check it, fix problems, then call seg_finish again with that review_code to save. Editing after the review requires a new review.",
			parameters: Type.Object({
				confidence: CONFIDENCE,
				reason: Type.String({ description: "Short reason for the confidence." }),
				review_code: Type.Optional(
					Type.String({ description: "Code printed in the latest final review; required to save." }),
				),
			}),
			async execute(_id, params) {
				return runOp("finish", asArgs(params));
			},
		}),
	];

	return (pi: ExtensionAPI) => {
		for (const tool of tools) {
			pi.registerTool(tool);
		}
		if (options.onProviderRequest) {
			const observe = options.onProviderRequest;
			pi.on("before_provider_request", (event) => {
				try {
					observe(event.payload);
				} catch {
					// Capture is diagnostic only; never disturb the agent loop.
				}
			});
		}
	};
}

/** pi CLI entry point: bind from `SEGD_URL` / `SEGD_CASE`. */
export default function piSegExtension(pi: ExtensionAPI): void {
	createSegExtension({
		segdUrl: process.env.SEGD_URL ?? SEGD_DEFAULT_URL,
		caseHandle: process.env.SEGD_CASE,
	})(pi);
}
