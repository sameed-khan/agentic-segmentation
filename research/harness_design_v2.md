# Harness design v2 and validation-phase handoff

Status: design agreed with Sameed on 2026-10-18. Infrastructure checks run the same day (results in §7 and §8).
This document is the **handoff for a fresh session** that will build the harness and run the
initial validation phase end to end. Read it top to bottom before touching code.

**Revision 2026-10-18 (Sameed):** Phase 2 (localization study) runs on the **30 design cases**, not the test split.
Its purpose at this stage is only to decide whether anatomical prompting or examples are needed; the full
localization study on the test split is run later. Consequences: no test-split download before Phase 3; no git
repo until the validation phase (Phases 0–2) is finished; Phase 2 conditions use the same 30 cases whose marks
are tuned in Phase 1, so coverage numbers are optimistic and n = 6 per histotype is a coarse go/no-go signal.
Agent thinking level: medium. Concurrency: 4 cases (matches `--max-sessions 4`). Multi-rater reference (future): voxel-wise majority vote.

Related documents:
- `abstract_draft.md`: the RSNA 2026 abstract (Sonnet 4.6, GT-centroid seed, median Dice 0.80).
- `research/potential_plan1.md`: earlier design brainstorm (code-first harness, set-of-marks, grounding).
- `research/HARNESS_DESIGN_RESEARCH.md`: longer literature memo behind potential_plan1.
- `eval/`: scripts and logs from the May 2026 experiment (results and transcripts were lost with the Azure VM).

---

## 1. What happened before, and what is wrong with it

Reconstructed from the surviving files in `eval/` (scripts, aggregate logs, three USAGE_GUIDE versions,
one smoke transcript). The `segctl` CLI/daemon source, all per-case results, and all transcripts lived on
the Azure VM and are gone. The harness is rebuilt from scratch.

| Date (2026) | Phase | Agent inputs | MSTT Dice | Vessel (TopBrain) Dice |
|---|---|---|---|---|
| 05-04 03:21 | Pilot 01, Sonnet 4.6, 40 cases | text target only, **no seed** | median 0.29, mean 0.39 (6/15 at 0.0) | median 0.00 (2 empty masks) |
| 05-04 03:25 | Guide rewrite | intensity table, per-histotype T2 appearance, vessel landmarks | – | – |
| 05-04 12:17 | Opus pilot | killed after 4 timeouts | – | – |
| 05-04 22:29 | Seed introduced | GT largest-component centroid + "do not view slices first; the seed is reliable … the seed eliminates the 'where is it' question" | – | – |
| 05-05 | Full run | seed + rewritten guide; T2 only | **median 0.80, mean 0.73**, n=166 | poor; dropped |
| 05-05 | `run_oneclick_baseline.py` | written; **no results survive** | – | – |

Findings that the new work must fix:

1. **The 0.80 is an oracle-seeded result.** The seed was the GT centroid and the prompt told the agent not to localize. The one surviving transcript (F02) is one click + `keep-largest-component` + views + stop.
2. **No one-click nnInteractive number exists**, so the agent's contribution beyond nnInteractive is unknown.
3. **The comparison against the published model was median vs mean.** The MSTT-199 per-histotype numbers are means (§4.3). Mean vs mean, the old run beats the published U-Net on 3/5 histotypes (F +0.12, N +0.02, V +0.14), loses on 2 (L −0.07, M −0.03). The abstract's "4 of 5, fibrous +0.22" is wrong.
4. **The 83% failure-flag recall counted "medium" as a flag.** The agent never used "low"; 97/166 were "medium".
5. **The abstract mentions scribble and polyline prompts**; the old guide only had point/box prompts and polygon edits.
6. **Wrong-structure errors on MSTT were labelled PARTIAL_COVERAGE** by `03_compute_metrics.py`.
7. **GT example images were built from evaluation cases** (`eval/examples/`). Dead code, no leakage occurred, but never revive them.
8. **Lipomatous had n=8** because only T2 masks were used; most lipomatous masks are on T1 (§4.1).

Recollections checked:
- *Pointing failure* (agent could not place a correct first click): confirmed in aggregate (pilot without seed), but
  semantic vs coordinate-serialization causes cannot be separated from surviving data. The old guide added `--snap`
  because clicks were "off by 5–20 px", which is evidence of coordinate error.
- *Anatomy-knowledge failure* (agent did not know what e.g. a fibrous tumor looks like on T2): **not supported** by
  surviving evidence. The localization study (§3, Phase 2) measures it directly.

---

## 2. Scientific framing

Question: when a general-purpose coding agent segments a 3D lesion through an interactive segmentation model,
which stage limits it?

- **Localization**: knowing what and where the lesion is.
- **Grounding**: turning that intent into an exact coordinate.
- **Delineation**: getting the boundary right.

Each harness component exists to serve one stage. Harness design choices are **not** ablated; they are fixed
by decision and described in the methods. The only planned comparisons are the hypotheses themselves:
raw coordinates vs set-of-marks (grounding), histotype label vs generic label (knowledge), agent vs
one-click nnInteractive vs published U-Net (overall).

Constraints: no RL or fine-tuning; the agent is pi (pi-coding-agent) with a project-local extension; the model
for the main run is `anthropic/claude-sonnet-5-5`.

---

## 3. Phase plan and gates

Thresholds are fixed now, before any data is seen. Change them only with Sameed's agreement, and record the change here.

### Phase 0: infrastructure (mostly done, §7–§8)
- Colab L4 + `nninteractive-server` + SSH port-forward: **verified**.
- Isolated pi launch with subscription auth and inline tool images: **verified**.
- MSTT design-split data downloaded to `data/mstt199/` (see §4.4): **done**.
- Colab idle/liveness check (33 min): **passed** (§7.5).
- Remaining: SDK batch runner (§8.3) smoke test.

### Phase 1: harness design on the 30 design cases
Build the harness (§5–§6), run the agent on the 30 design cases **without any seed**, inspect failures, fix
harness bugs and usability problems, repeat. The design cases are a development set and are never reported as results.

Freeze criteria (harness v1 is frozen when all hold on a full pass over the 30 design cases):
- 30/30 cases finish without harness errors (tool exceptions, coordinate bugs, crashed sessions).
- Mark coverage: at least one coarse-or-fine blob/region mark lies inside the GT lesion in ≥ 90% of design cases (offline diagnostic; GT never reaches the agent).
- No open bugs in the failure log tagged `harness`.

Record each pass in `eval/v2/design_log.md`: date, harness commit, per-case Dice, first-positive-click-inside-lesion, failure tag, notes.

### Phase 2: localization study (test split, single-turn calls)
Purpose: decide whether pointing works and whether anatomical prompting or a highlighted example is needed.
Runs on the **test** split (169 cases) with the frozen rendering and mark code. No agent loop: one rendered input, one answer per call.

Input: the axial slice through the GT lesion's largest cross-section (in-plane pointing only; slice choice is tested in Phase 3).
Conditions (all with `anthropic/claude-sonnet-5-5`):

| ID | Interface | Target text |
|---|---|---|
| R | raw: image with pixel rulers, answer `(x, y)` | histotype text, e.g. "fibrous soft-tissue tumor" |
| S | set-of-marks: answer a mark ID | histotype text |
| S-gen | set-of-marks | "soft-tissue mass" (label-swap knowledge test) |
| S-desc | set-of-marks | histotype text + 1–2 sentence T1/T2 appearance description written from the literature |
| S-ex | set-of-marks | histotype text + one highlighted example (GT outline) from a **design** case of the same histotype |

S-desc and S-ex are run only if S fails the gate.

Metrics: hit rate (chosen point inside GT), distance to GT in mm, mark coverage (does any mark lie inside the GT), selection accuracy given coverage, all per histotype.

Gate to Phase 3:
- S hit rate ≥ 0.80 overall **and** S − R ≥ 0.10. Then proceed with S.
- If S < 0.80: check coverage. If coverage < 0.90, fix the mark generator (on design cases only) and rerun.
  If coverage ≥ 0.90 but selection fails, run S-desc and S-ex and adopt whichever passes. If none passes, stop and discuss.
- S vs S-gen answers the knowledge question. If S ≈ S-gen, the model is not using histotype knowledge.

### Phase 3: full agent run on MSTT test split
- Agent: frozen harness v1, Sonnet 5.5, no seed, target text = histotype description, sequence = the annotated one (T1 or T2).
- Also run **one-click nnInteractive at the GT centroid** (no LLM; cheap) on the same cases, to quantify how much nnInteractive alone delivers.
- Primary comparison: per-histotype **mean** Dice vs published U-Net (§4.3). Report medians as secondary.
- Gate to Phase 4: mean Dice ≥ 0.681 (published U-Net average) and higher than the U-Net on at least 3 of 5 histotypes.

### Phase 4: expansion (scoped later)
More datasets (one released after 2026-06-30 for leakage defence), more models, interventions motivated by the
Phase 1 and 3 failure logs (e.g. adaptive brush). Candidate datasets are in the conversation summary at the end of this document; their release dates and baselines must be verified before use.

---

## 4. Data: MSTT-199

### 4.1 Facts (verified 2026-10-18 from the Kaggle file listing)
- Kaggle dataset `reasat/mstt-199`, licence CC0, 89.6 GB. Public anonymous API download works:
  `https://www.kaggle.com/api/v1/datasets/download/reasat/mstt-199/<url-encoded path>`.
- Only the **isotropic** version has masks. `reasat/mstt-199-aniso` has images only. Use isotropic (as the old run did).
- Files are uncompressed NIfTI, **float64**, 1 mm isotropic. Image median 124 MB, max 594 MB. Masks are also float64.
- 199 cases: F 40, L 40, M 40, N 40, V 39. Each case has masks on exactly one sequence:
  lipomatous mostly **T1** (31 T1, 9 T2), all other histotypes **T2**. Images exist for both T1 and T2 for every case.
- **15 cases have three-rater masks** (suffix `_J`, `_K`, `_N`): V 6, F 4, L 3, M 1, N 1. They are excluded from the
  design split and kept in the test split. Use them for an inter-rater ceiling. Decide before Phase 3 which mask is the
  reference for these cases (proposal: majority vote of the three, reported alongside the pairwise inter-rater Dice).
- Paired T1 images were not registered to T2 in this dataset as far as we know; do not assume voxel correspondence between sequences without checking.

### 4.2 Split
`eval/v2/mstt_cases.csv`, built by `eval/v2/build_design_manifest.py` (seed 2026):
- design: 30 cases, 6 per histotype (L: 5 T1 + 1 T2), no multi-rater cases.
- test: 169 cases (F 34, L 34, M 34, N 34, V 33).
Columns: `case, histotype, sequence, multi_rater, image_kaggle, masks_kaggle, image_mb, split`.

### 4.3 Published baselines (MSTT-199 paper, arXiv 2409.03110; numbers supplied by Sameed from the paper)

| Model | Input | Train | Test STS | Test MSTT-199 |
|---|---|---|---|---|
| U-Net | MRI | MSTT-199 | 0.79 | 0.68 |
| LiteMedSAM | MRI | MSTT-199 | 0.80 | 0.67 |
| Multi-Branch U-Net | MRI, PET | STS | 0.77 | – |

Per tissue type (U-Net, **mean** Dice; 0.681 is the unweighted mean of the five):

| Fibrous | Fat | Myxoid | Nerve | Vascular | Average |
|---|---|---|---|---|---|
| 0.507 | 0.744 | 0.823 | 0.748 | 0.584 | 0.681 |

Per site (mean Dice; count of cases): thigh 0.774 (103), leg 0.505 (25), glute 0.711 (19), forearm 0.715 (13),
arm 0.596 (10), hand 0.381 (4), shoulder 0.485 (4), foot 0.351 (3), neck 0.411 (3), pelvis 0.595 (3),
chest wall 0.415 (3), abdominal wall 0.020 (2), axilla 0.276 (1), back 0.325 (1), flank 0.148 (1). Total 195.

Caveats for the presentation: the baseline is their U-Net (SE-ResNeXt-50 encoder per the repo), not nnU-Net; per-case
predictions are not available, so comparisons are unpaired against group means; our test split excludes the 30 design cases.
The site table shows the U-Net fails at less common sites; if per-case site metadata can be found (paper supplement or repo), a site-stratified comparison is worth one slide.

### 4.4 Download
`uv run eval/v2/download_mstt.py [design|test|all] [dest]` → `data/mstt199/{image,mask}/`. Idempotent.
Design split downloaded on 2026-10-18: 60 files (30 images + 30 masks), 9.9 GB, in `data/mstt199/`.
Full dataset is 89.6 GB; local disk had 120 GB free on 2026-10-18, so download the test split only when needed
(or download it directly on the rental GPU box).

---

## 5. Harness design

### 5.1 Principles
1. **No raw coordinates in the default action space.** Spatial arguments are mark IDs, slice indices, or component references. Raw pixel/voxel coordinates exist only behind the config flag `pointing: raw` for the Phase 2 comparison.
2. **Tools return images inline** (pi tool results carry image content; verified §8.1). No separate `read` turn per view.
3. **Every mutating tool returns a compact observation**: volume (mL), number of components, slice extent per axis, change from parent version (added/removed mL), plus an outline render of the changed region.
4. **Immutable mask versions** kept by the daemon. Undo/checkout are cheap; every version is saved so Dice can be computed for every intermediate state afterwards.
5. **GT never reaches the agent process.** The agent has only `seg_*` tools: no bash, read, write, or edit. GT is read only by offline scoring scripts.
6. **Everything is logged** (§5.7). Transcripts were lost last time; the runner copies all artifacts into the results directory.

### 5.2 Components
- **`segd` (Python daemon, local)**: holds per-case state: image array, spacing/affine, the nnInteractive remote session, mask versions, mark cache, render engine. Exposes a local HTTP API on 127.0.0.1 (one process, many cases). Talks to nnInteractive through `nnInteractiveRemoteInferenceSession` (identical client for Colab and the rental box).
- **pi extension (TypeScript)**: registers `seg_*` tools; each tool is a thin call to `segd`; returns text + PNG.
- **`segctl` (thin CLI)**: same `segd` API for debugging and human use. Keeps the "custom CLI" description in the abstract accurate.
- **Runner (TypeScript, pi SDK)**: batch execution, one pi session per case (§8.3).
- **Scorer (Python)**: offline Dice/HD95/NSD per version, localization metrics, failure tagging.

### 5.3 Tool set

| Tool | Arguments | Effect / returns |
|---|---|---|
| `seg_open` | (none; case bound by runner) | Case card (shape, spacing, sequence, orientation, target text) + orthogonal overview with orientation letters (R/L, A/P, S/I) and a 10 mm scale bar |
| `seg_sheet` | axis, slice range, n | Contact sheet labelled with slice indices (navigation) |
| `seg_view` | axis, slice \| "mask", crop (full \| mask \| bbox of marks), overlay (none \| outline \| fill), marks (none \| coarse \| fine \| edges) | Clean image and marked image side by side |
| `seg_point` | mark IDs, polarity | nnInteractive point prompt(s) |
| `seg_lasso` | ordered mark IDs (≥3, same slice), polarity | Polygon → nnInteractive lasso (2D, propagated to 3D by the model) |
| `seg_scribble` | ordered mark IDs, polarity | Polyline → nnInteractive scribble |
| `seg_box` | two mark IDs (same slice) | 2D box prompt (public weights support 2D boxes only) |
| `seg_paint` | ordered mark IDs, add \| erase, slice range | Direct raster edit, no model |
| `seg_refine` | (none) | Send current mask via `add_initial_seg_interaction` so nnInteractive refines a painted mask |
| `seg_edit` | op: keep_component(mark) \| remove_component(mark) \| fill_holes \| close \| open \| dilate \| erode; radius_mm | Morphology/component edits |
| `seg_undo` / `seg_checkout` | version id | Version control |
| `seg_finish` | confidence (high \| medium \| low), reason | Saves final mask; ends the case |

nnInteractive API facts (v2.6.0 server): `add_point_interaction`, `add_bbox_interaction` (2D only with public weights),
`add_scribble_interaction` and `add_lasso_interaction` (pass a small crop with `interaction_bbox`; much faster than
full-volume masks), `add_initial_seg_interaction`, `reset_interactions`, single-level `undo`. The server returns only
the changed region, blosc2-compressed.

### 5.4 Marks (set-of-marks pointing)
Computed once per volume at `seg_open` and cached on disk. Parameters are tuned on design cases only, then frozen.

1. Normalize intensity: clip to 0.5–99.5 percentile inside the body mask (Otsu foreground, largest component, holes filled).
2. **Blob marks** (peaks and valleys): Laplacian-of-Gaussian at several scales (σ ≈ 2, 4, 8, 16 mm), on a 2 mm downsampled volume for speed; keep local extrema of both signs above a response threshold; store centre and scale.
3. **Region marks** (centres of homogeneous areas): SLIC supervoxels sized ~15–25 mm; for each, the point furthest from its boundary (Euclidean distance transform).
4. **Edge marks**: computed per view on the displayed 2D slice (gradient-magnitude maxima / Canny), with IDs valid only for that render. Used as lasso/scribble vertices and negative-click candidates.
5. Thin blob + region marks with a minimum spacing: coarse ≈ 15 mm, fine ≈ 5 mm. Keep only marks inside the body.
6. Blob and region marks have **global 3D IDs**, stable across axial/coronal/sagittal views. A slice shows a mark if it lies within about ±(scale/2, minimum 2 mm) of the slice.
7. Display: numbered labels with a contrasting halo; a different glyph per mark type; at most ~40–60 labels per panel; cropping a view reveals the fine marks inside the crop (coarse-to-fine).
8. Offline diagnostic: mark coverage of the GT lesion (§3 Phase 1 criterion).

### 5.5 Rendering
- Panels about 768 px on the long side; clean + marked side by side ≈ 1536 px wide, within Claude's image size limit (~1568 px long edge) so labels are not downsampled.
- Window: percentile-based per sequence; no user control in v1.
- Radiological display convention stated in the case card; orientation letters on every panel; scale bar; slice index in the corner.
- Overlay: outline by default (fill hides the boundary).

### 5.6 Coordinates
- Internal space: voxel indices of the loaded isotropic array, with the NIfTI affine kept for export. One canonical orientation (RAS+ via nibabel `as_closest_canonical`) applied at load.
- Each mark stores voxel index and mm position; the agent never sees either in `pointing: marks` mode.
- Volumes are sent to nnInteractive as float16 (verified to work). If a compressed upload exceeds limits or is slow, crop to the body bounding box and keep the offset in `segd` (lossless for segmentation).

### 5.7 Logging (per case, under `results/<run_id>/<case>/`)
- `session.jsonl`: the pi session file (all messages, tool calls, tool results incl. images).
- `trajectory.jsonl`: every `segd` operation: args, resolved voxel coordinates, timings (render, nnInteractive, total), parent/child version ids, observation summary.
- `renders/*.png`: every image shown to the agent.
- `versions.npz`: all mask versions (uint8, compressed).
- `final_mask.nii.gz`, `meta.json`: model id, thinking level, pi version, auth-shim version, harness commit, nnInteractive client/server versions and model id, system prompt hash, wall time, tokens, cost.
- Back up `results/` off the GPU box after every batch.

### 5.8 Prompts
- System prompt: short; describes tools, the marks convention, the observation format, budgets, and the stop rule. No GT examples. No instruction to trust any seed (there is none).
- Task message: case card fields + target text (e.g. "fibrous soft-tissue tumor on T2-weighted MRI").
- Budgets (initial; tune in Phase 1): ≤ 40 tool calls, ≤ 15 nnInteractive prompts, 10 min wall time.

---

## 6. Repository scaffold

```
harness/
  checks/            # done: auth_check.sh, ping_ext.ts, ping.png, remote_nni_check.py
  colab/             # done: start_server.py; logs/ (gitignore)
    up.sh            # TODO: colab new + install + start server + ssh forward (§7.4)
    down.sh          # TODO: colab stop + ssh -O exit
  server/            # TODO: segd (uv project)
    pyproject.toml   #   deps: numpy, nibabel, scipy, scikit-image, pillow, blosc2, fastapi/uvicorn, nninteractive-client
    segd/
      app.py         #   HTTP API, one CaseState per open case
      case_state.py  #   image, affine, versions, nnInteractive remote session
      marks.py       #   §5.4
      render.py      #   §5.5
      ops.py         #   point/lasso/scribble/box/paint/refine/edit/undo/checkout
      observe.py     #   compact observation after each mutation
      log.py         #   trajectory.jsonl, renders, versions
    segctl.py        #   thin CLI client
    tests/           #   phantom tests: coordinate round-trips, mark IDs stable across views, version DAG
  extension/         # TODO: pi extension (TypeScript)
    index.ts         #   registerTool for each seg_* tool → segd HTTP; returns text + image
  runner/            # TODO: pi SDK batch runner
    package.json     #   pins @earendil-works/pi-coding-agent to an exact version (1.2.0 on 2026-10-18)
    run_cases.ts     #   §8.3
  prompts/
    system.md        # TODO
    task.md          # TODO: case-card template
  study/             # TODO: Phase 2 localization study (single-turn calls)
eval/
  v2/                # done: mstt_cases.csv, build_design_manifest.py, download_mstt.py, kaggle file listing
    score.py         # TODO: Dice/HD95/NSD per version, localization metrics, failure tags
    design_log.md    # TODO: Phase 1 pass log
data/mstt199/        # downloaded data (not committed)
results/             # run outputs (not committed; back up)
```

Initialize a git repository at the project root before writing harness code (the project currently is not a git repo),
with `data/`, `results/`, `harness/colab/logs/` in `.gitignore`. Record the harness commit in every `meta.json`.

---

## 7. GPU: Colab L4 via the Colab CLI (verified 2026-10-18)

### 7.1 What was verified
- `google-colab-cli` 0.7.4 installed with `uv tool install google-colab-cli`; OAuth done once (`colab --auth=oauth2`, the default; token cached under `~/.config/colab-cli/`).
- Account: Google AI Pro (premium Colab included since 2026-09-22). `colab usage` showed **200 compute units**; an L4 runtime uses **1.54 units/h** (≈130 h).
- `colab new -s nnint --gpu L4`: ready in ~3 s. VM: NVIDIA L4 23 GB, driver 580, Python 3.13, torch 2.11.0+cu130 preinstalled.
- `colab install -s nnint nnInteractive`: 38 s, nnInteractive 2.6.0 (torch untouched).
- `colab exec -s nnint -f harness/colab/start_server.py`: starts `nninteractive-server --model nnInteractive_v1.0 --host 127.0.0.1 --port 1527 --max-sessions 4` and blocks streaming its log, so the kernel stays BUSY. Server ready ~60 s after launch (model download + torch.compile warm-up).
- SSH over WebSocket with OpenSSH port-forwarding works: `ssh -O forward -L 1527:127.0.0.1:1527 colab-nnint`, then `curl http://127.0.0.1:1527/healthz` → `{"ok":true}` locally.
- Round trip with `harness/checks/remote_nni_check.py` (synthetic phantom, float16):

| Volume | Voxels | Packed upload | `set_image` | Point prompt |
|---|---|---|---|---|
| 231×231×343 (median-sized MSTT) | 18.3 M | 14.7 MB | 5.2 s | 0.6–0.8 s |
| 300×300×800 (≈ largest MSTT) | 72.0 M | 57.8 MB | 16.6 s | 0.6–0.7 s |

- Idle/liveness: see §7.5.

### 7.2 Local setup (outside the project; deliberately **not** recorded in personal-setup)
- `~/.ssh/id_ed25519_colab` (Colab rejects RSA keys).
- `~/.ssh/config` entry:
```
Host colab-nnint
  ProxyCommand /home/sameed/.local/bin/colab ssh --proxy-mode -s nnint -i ~/.ssh/id_ed25519_colab
  User root
  IdentityFile ~/.ssh/id_ed25519_colab
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
  ControlMaster auto
  ControlPath ~/.ssh/cm/%r@%h
  ControlPersist 4h
  ServerAliveInterval 30
```
- Only one `colab ssh` connection per runtime is allowed (HTTP 429 otherwise). `ControlMaster` multiplexes the shell and the port-forward over that one connection.

### 7.3 Facts and limits
- The VM stays alive based on kernel activity; Pro does **not** include Ultra's "uninterrupted background execution". Keep the server inside a blocking `colab exec`.
- While `start_server.py` runs, the kernel is busy, so further `colab exec` calls queue behind it. Use `ssh colab-nnint '<cmd>'` for other commands on the VM.
- The Colab session URL and VM change on every `colab new`; the SSH host alias and local port stay the same.
- Upload bandwidth from home is the main per-case cost (≈5–17 s per case on 2026-10-18).
- `colab stop -s nnint` releases the VM and stops billing. Always stop when idle.

### 7.4 Bring-up sequence (to be scripted as `harness/colab/up.sh`)
```bash
colab new -s nnint --gpu L4
colab install -s nnint nnInteractive
nohup colab exec -s nnint -f harness/colab/start_server.py > harness/colab/logs/server_exec.log 2>&1 &
# wait until the log shows "Uvicorn running on http://127.0.0.1:1527"
ssh -fN colab-nnint                       # opens the ControlMaster connection
ssh -O forward -L 1527:127.0.0.1:1527 colab-nnint
curl -s http://127.0.0.1:1527/healthz     # {"ok":true}
```
Tear-down: `ssh -O exit colab-nnint; colab stop -s nnint`.

### 7.5 Idle/liveness test
Last server traffic 2026-10-18 10:42 EDT; kernel kept busy by `start_server.py`. At 11:15 EDT (33 min with no
requests) the VM was still assigned, status BUSY, and `/healthz` answered through the forward. **Passed.**
Longer idle periods (hours) are untested; stop the VM when not running cases anyway.
The whole Phase 0 test (≈45 min of L4) used 0.89 compute units (200 → 199.11). The VM was stopped afterwards.

### 7.6 Fallbacks and later
- If the SSH route fails: run `cloudflared tunnel --url http://localhost:1527` on the VM (via `ssh` or `colab exec`), use the printed `trycloudflare.com` URL, and set `--api-key` on the server. Limits: 100 MB per request, ~100 s response timeout, URL changes per run. ngrok's free tier (1 GB/month, uploads included) and Hobbyist (5 GB) are too small for this workload.
- Rental GPU box for the full run: `docker run` the official nnInteractive server image (or `pip install nnInteractive`), bind to 127.0.0.1, run `segd` and the runner on the same box. Client code is unchanged.
- The ITK-SNAP DLS server (`itksnap-dls`) is **not** used: it uploads and returns full gzip volumes for each scribble/lasso and has no undo or initial-mask refinement.

---

## 8. Running pi for the experiment

### 8.1 Verified: isolated launch with subscription auth (2026-10-18)
`harness/checks/auth_check.sh`:
```bash
pi -ne -e harness/checks/ping_ext.ts \
   -e ~/.pi/agent/npm/node_modules/@gotgenes/pi-anthropic-auth \
   --no-mcp -ns -np -nc --no-themes -na --no-session \
   --tools 'seg_*' --model anthropic/claude-sonnet-5-5 \
   --system-prompt "..." -p "..."
```
Result: the model saw exactly one tool (`seg_ping`), called it, and correctly described the returned inline PNG
(red circle, blue background, digit 7). JSON mode confirmed `provider=anthropic, model=claude-sonnet-5-5`. ~6 s.

**The auth shim is required.** Without `@gotgenes/pi-anthropic-auth`, Anthropic rejects the request:
`400 invalid_request_error: "Third-party apps now draw from your extra usage, not your plan limits."`
So "no extensions" for the experiment means: our extension **plus** the auth shim. The shim rewrites the system prompt
for OAuth requests, so log the system prompt actually sent (provider payload event) and pin the shim version (3.4.2 on 2026-10-18).

A harmless warning appears from the user's `settings.json` (`enabledModels` pattern for opencode-go); `--model` overrides the default model.

### 8.2 Interactive debugging wrapper (Phase 1)
`harness/bin/pi-seg` (TODO): the command above with `-e harness/extension`, `--tools 'seg_*'`, the project system prompt,
and without `-p`. The extension is never placed in `~/.pi/agent/extensions`, a pi package list, or a project `.pi/extensions`
folder, so the user's normal pi never loads it.

### 8.3 Batch runner (pi SDK; TODO, smoke-test first)
`harness/runner/run_cases.ts`, with its own `package.json` pinning `@earendil-works/pi-coding-agent` to an exact version
(so `pi update` on the machine cannot change the experiment). Design:
- `ModelRuntime.create({ authPath: "~/.pi/agent/auth.json", modelsPath: "<project>/harness/runner/models.json" })` uses the user's OAuth token in place (refreshes write back to the same file; no copy can drift).
- Custom `ResourceLoader`: extensions = [`harness/extension`, the auth shim]; no skills, prompt templates, themes, or context files; system prompt from `harness/prompts/system.md`.
- `SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: true, maxRetries: 2 } })`.
- `tools: ["seg_*"]` only (resolve to explicit names).
- One `AgentSession` per case with a persistent `SessionManager` writing into `results/<run_id>/<case>/`.
- Run cases concurrently **inside one Node process sharing one `ModelRuntime`**, so only one OAuth refresh happens at a time (parallel processes refreshing the same token can invalidate each other).
- Per-case wall-time limit via `session.abort()`; record tokens/cost from `message_end` usage.
- Smoke test before Phase 1 runs: one case with `ping_ext.ts` in place of the real extension, confirming auth, tool isolation, and session file output. Example code: `~/.pi/agent/install/releases/1.2.0/node_modules/@earendil-works/pi-coding-agent/examples/sdk/12-full-control.ts`.

### 8.4 Subscription limits
Running on the Claude subscription consumes 5-hour usage windows; heavy runs can be throttled. Use the subscription for
Phase 1–2. Decide before Phase 3 whether to switch to an API key (`modelRuntime.setRuntimeApiKey("anthropic", key)`), which also improves reproducibility.

---

## 9. Handoff: task list for the execution session

Do these in order. Each has an acceptance check. Ask Sameed before changing gate thresholds or the phase plan.

1. **Repo hygiene.** `git init` at the project root; `.gitignore` for `data/`, `results/`, `harness/colab/logs/`, `node_modules/`, `__pycache__/`. First commit.
2. **Data.** The design download finished (60 files in `data/mstt199/`). Write a loader that opens each case and checks image/mask shape and affine match; report any mismatch.
3. **Upload-size pre-check.** For the 30 design cases, compute the float16 blosc2 packed size (`nnInteractive.inference.remote.serialization.pack_array`). Flag any > 90 MB.
4. **Colab scripts.** Write `harness/colab/up.sh` and `down.sh` from §7.4. Acceptance: `up.sh` ends with `{"ok":true}`.
5. **One-click baseline on design cases.** Script: GT centroid (largest component, snapped inside the mask) → one point prompt → Dice. No LLM. Gives an early sense of how much nnInteractive alone does on these cases and exercises the remote path on real data.
6. **`segd` core + phantom tests.** CaseState, versions, coordinate handling, nnInteractive ops, observation, logging. Tests on synthetic phantoms: a mark ID resolves to the same voxel from all three views; lasso/point on a phantom lesion gives Dice > 0.9.
7. **Marks + rendering.** §5.4–5.5. Acceptance: renders legible at the target size; mark coverage ≥ 0.90 on design cases.
8. **Extension + `pi-seg` wrapper.** Acceptance: interactive session on one design case end to end.
9. **SDK runner.** Smoke test (§8.3), then a 3-case batch.
10. **Phase 1 passes** on the 30 design cases until the freeze criteria (§3) hold. Keep `eval/v2/design_log.md`.
11. **Phase 2 localization study** on the test split. Report against the gate.
12. **Stop and report to Sameed** before starting Phase 3.

Always `colab stop -s nnint` when not actively running cases.

---

## 10. Open questions and risks
- Reference mask for the 15 three-rater cases (proposal in §4.1).
- Whether per-case tumour site metadata exists for MSTT-199 (needed for a site-stratified comparison).
- Sonnet 5.5 behaviour with dense mark labels; tune label count on design cases.
- Subscription throttling during long batches (§8.4).
- Colab runtime reclamation during long idle periods (§7.5).
- Phase 4 dataset candidates (unverified; found by a DeepSeek research subagent on 2026-10-18; check dates and baselines before use):
  Spinal-Multiple-Myeloma-SEG (Sci Data 2026-08-11, CC BY 4.0, nnU-Net weights on Zenodo; spine lesions, CT),
  BEAMSTER (Sci Data 2026-07-01, small brain metastases, CE-T1 MRI; check for an earlier preprint),
  TopAneu 2026, ImageCAS-X (labels new, images from 2023), 3DLAND masks, FLARE 2026 post-June sub-datasets.
  No public post-cutoff 3D bone-marrow-oedema dataset was found.
