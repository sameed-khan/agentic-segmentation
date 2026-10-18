/**
 * Smoke-test extension for the isolated experimental pi launch.
 * Registers one tool, `seg_ping`, that returns text + an inline PNG.
 * Verifies: (1) explicit -e extension loads under -ne, (2) tool results can
 * carry images that the model actually sees (no separate `read` turn).
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const here = dirname(fileURLToPath(import.meta.url));

const pingTool = defineTool({
	name: "seg_ping",
	label: "Seg ping",
	description: "Health check. Returns a short text and a test image.",
	parameters: Type.Object({}),
	async execute() {
		const data = readFileSync(join(here, "ping.png")).toString("base64");
		return {
			content: [
				{ type: "text", text: "pong" },
				{ type: "image", data, mimeType: "image/png" },
			],
			details: {},
		};
	},
});

export default function (pi: ExtensionAPI) {
	pi.registerTool(pingTool);
}
