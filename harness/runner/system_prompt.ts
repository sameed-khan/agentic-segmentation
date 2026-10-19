/**
 * System prompt file selection for the segmentation harness runner.
 *
 * An explicit `--system-file` always wins (baseline-prompt ablations), even when its
 * resolved path equals the default: equality with the default path must not re-enable
 * the v3 variant. Without an explicit flag, a v3 run uses the v3 variant, everything
 * else the default.
 */
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(HERE, "..", "..");
const HARNESS_DIR = join(PROJECT_ROOT, "harness");

/** Default system prompt (v2 path). */
export const DEFAULT_SYSTEM_FILE = join(HARNESS_DIR, "prompts", "system.md");
/** Harness v3 system prompt (adds the seg_locate / second-reader paragraph). */
export const DEFAULT_SYSTEM_V3_FILE = join(HARNESS_DIR, "prompts", "system_v3.md");

export function selectSystemFile(
	systemFileSupplied: boolean,
	systemFile: string,
	v3On: boolean,
): string {
	return systemFileSupplied
		? systemFile
		: v3On
			? DEFAULT_SYSTEM_V3_FILE
			: DEFAULT_SYSTEM_FILE;
}
