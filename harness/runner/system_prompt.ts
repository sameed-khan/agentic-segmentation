/**
 * System prompt file selection for the segmentation harness runner.
 *
 * An explicit `--system-file` always wins (baseline-prompt ablations), even when its
 * resolved path equals the default: equality with the default path must not re-enable
 * the v3 variant. Without an explicit flag: a v3 run uses the v3 variant, a non-MSTT
 * dataset (Phase 4) the multi-lesion variant, everything else the default.
 *
 * v3 + non-MSTT has no combined variant (seg_locate paragraph + multi-lesion rules):
 * running that combination requires an explicit --system-file, so selection throws
 * instead of silently picking the wrong rule set.
 */
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(HERE, "..", "..");
const HARNESS_DIR = join(PROJECT_ROOT, "harness");

/** Default system prompt (v2 path, single lesion). */
export const DEFAULT_SYSTEM_FILE = join(HARNESS_DIR, "prompts", "system.md");
/** Harness v3 system prompt (adds the seg_locate / second-reader paragraph). */
export const DEFAULT_SYSTEM_V3_FILE = join(HARNESS_DIR, "prompts", "system_v3.md");
/** Phase 4 multi-lesion system prompt (all lesions of the target; no MSK rim/fascia rule). */
export const DEFAULT_SYSTEM_MULTI_FILE = join(HARNESS_DIR, "prompts", "system_multi.md");

export function selectSystemFile(
	systemFileSupplied: boolean,
	systemFile: string,
	v3On: boolean,
	dataset = "mstt",
): string {
	if (systemFileSupplied) return systemFile;
	if (v3On) {
		if (dataset !== "mstt") {
			throw new Error(
				`no combined system prompt exists for v3 + dataset ${dataset}: pass an explicit --system-file ` +
					"(the multi-lesion and seg_locate/second-reader paragraphs have not been merged)",
			);
		}
		return DEFAULT_SYSTEM_V3_FILE;
	}
	return dataset === "mstt" ? DEFAULT_SYSTEM_FILE : DEFAULT_SYSTEM_MULTI_FILE;
}
