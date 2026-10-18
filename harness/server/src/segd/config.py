"""Central tunables for segd (single source of truth for sizes, budgets, endpoints).

No behavioral logic lives here. Values are read through the module at call time
(``segd.config.X``) so tests can monkeypatch them.
"""

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
