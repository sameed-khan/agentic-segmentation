"""Central tunables for segd (single source of truth for sizes, budgets, endpoints).

No behavioral logic lives here. Values are read through the module at call time
(``segd.config.X``) so tests can monkeypatch them.
"""
from pathlib import Path

UPLOAD_MAX_PACKED_BYTES = 60 * 1024 * 1024   # upload the full float16 volume up to this packed size...
CROP_MARGIN_VOX = 5                          # ...else crop to the body bbox plus this margin (CONTRACTS §4)
INTERACTION_BBOX_MARGIN = 8                  # in-plane margin around a drawn lasso/scribble bbox, kept only
                                             # where it stays inside the uploaded crop (ops._interaction_box_for)
EDIT_DEFAULT_RADIUS_MM = 2.0                 # morphology radius when seg_edit omits radius_mm (documented in the tool)
VERSIONS_NPZ_EVERY = 5                       # flush versions.npz every N versions and at finish

HOST = "127.0.0.1"
PORT = 8765

DEFAULT_BUDGETS = {"max_tool_calls": 40, "max_model_prompts": 15}

AXES = ("axial", "coronal", "sagittal")

# Display orientation (CONTRACTS §1). Arrays are [i, j, k]; +i = patient Right, +j = Anterior,
# +k = Superior. A displayed 2D slice is the raw slice transposed and flipped on both display axes
# so that row 0 / column 0 are the *larger* index of the row/col axis:
#   axial    (k fixed): rows = j (anterior at top), cols = i (patient right at left)
#   coronal  (j fixed): rows = k (superior at top), cols = i (patient right at left)
#   sagittal (i fixed): rows = k (superior at top), cols = j (anterior at left)
FIXED_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}
ROW_AXIS = {"axial": 1, "coronal": 2, "sagittal": 2}
COL_AXIS = {"axial": 0, "coronal": 0, "sagittal": 1}

# seg_* tools that send a prompt to nnInteractive (subject to the model-prompt budget).
# Every seg_* call counts against the tool-call budget (see ops.run_op); when that budget
# is exhausted only `finish` is accepted.
MODEL_PROMPT_OPS = frozenset({"point", "lasso", "scribble", "box", "refine"})

# Candidate corrections (segd.candidates; benchmark results/candidate_bench/summary.md
# chose 'intensity' as the best precision/recall-per-second source, max 5 per state).
CANDIDATE_METHOD = "intensity"
CANDIDATE_MAX_N = 5
CANDIDATE_REVIEW_MAX_ACCEPT_ROUNDS = 3   # after this many seg_accept calls a review stops offering candidates

# Harness v3 (optional local-model components; research/harness_v3_methodology.md §3 Study C).
# ALL OFF by default: when the case's v3 flags are absent, every code path behind them is
# skipped and behaviour is byte-identical to v2.
V3_CHECKPOINTS = ("localize", "review", "candidates")
V3_POINTER_DEFAULT_URLS = {"gui_actor": "http://127.0.0.1:8092", "molmo2": "http://127.0.0.1:8093"}
V3_MEDGEMMA_DEFAULT_URLS = {"q4": "http://127.0.0.1:8091", "bf16": "http://127.0.0.1:8094"}
V3_LOCATE_K = 5                          # points requested from the pointer service
V3_POINTER_TIMEOUT_S = 60.0
V3_MEDGEMMA_TIMEOUT_S = 180.0            # per-call cap: a single call must never hang forever
V3_MEDGEMMA_GROUP_TIMEOUT_S = 150.0      # ABSOLUTE MedGemma wall-clock deadline per OP
                                         # (perf_counter at run_op ENTRY, before ANY
                                         # checkpoint work: pre-hook rendering and the
                                         # checkpoint calls share it), shared by ALL
                                         # checkpoint calls of the op: the extension aborts
                                         # one segd op after SEGD_OP_TIMEOUT_MS (180 s) and a
                                         # finish op runs the review plus up to five candidate
                                         # calls sequentially, so the group stays 30 s below
                                         # that deadline (reserved for render saving, trajectory
                                         # logging and response transport); checkpoints that
                                         # cannot finish get their unavailable note instead
V3_MEDGEMMA_RESPONSE_RESERVE_S = 15.0    # seconds withheld from every checkpoint call's
                                         # allowance: after the last MedGemma reply there must
                                         # still be time to format notes, save renders and log
                                         # the trajectory, so the whole op returns before the
                                         # extension's HTTP deadline (per-op budget above)
V3_MEDGEMMA_MAX_TOKENS = 256
# Single-sourced helpers that live outside the segd package (Phase 5 local models):
_V3_REPO_ROOT = Path(__file__).resolve().parents[3]
V3_MEDGEMMA_CLIENT_FILE = _V3_REPO_ROOT / "local_models" / "medgemma" / "medgemma_client.py"
V3_PROMPT_DIR = (_V3_REPO_ROOT / "prompts" / "medgemma")
