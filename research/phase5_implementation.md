# Phase 5 implementation: local models + Studies A/B — build and run book

**Built 2026-10-19** (branch `phase5-v3`, worktree `agseg-p5`). Everything here is code +
tests with **mocked backends**; the real model runs happen later on the A100 80 GB box.
Read `research/harness_v3_methodology.md` first (Sameed's decisions since it was written:
BiomedParse v2 and Qwen3-VL-8B are **skipped**; Study C agent = Sonnet 5.5 high, confirmed
with Opus 5.5 medium; Phase 5 runs on an A100 80 GB, so **full precision is the default**
and MedGemma Q4_K_M is kept as the one deployment-realism arm).

## 1. What exists (file map)

```
harness/local_models/                    own uv projects (pinned versions per the research notes)
  pointd_gui_actor/
    server.py          POST /point {image_png_b64, query, k, verify} -> {points:[{x,y,score,rank}], raw,
                       latency_ms, model, version, image_w, image_h, verifier}; GET /healthz. Port 8092.
                       Wraps the OFFICIAL GUI-Actor-2B inference (microsoft/GUI-Actor repo checkout,
                       transformers==4.51.3, qwen-vl-utils==0.0.8, accelerate==1.1.1; BF16; attn sdpa
                       default, flash_attention_2 behind an extra). Optional verifier flag reranks
                       the connected-region candidates with GUI-Actor-Verifier-2B (official
                       red-circle protocol per verifier/verifier_model.py: Qwen2VLForConditional-
                       Generation + AutoProcessor, circle radius 7 / width 3 / 500 px half-size
                       crop, score = P(True)/(P(True)+P(False)); a verify=true request FAILS
                       explicitly when the verifier could not be loaded). --fake mode:
                       deterministic dummy points, torch never imported.
    pyproject.toml     pins + [tool.uv] package=false (app-style project)
  pointd_molmo2/
    server.py          same /point API. Port 8093. Molmo2-8B via transformers==4.57.1,
                       trust_remote_code=True, EXPLICIT torch_dtype=bfloat16 (checkpoint is FP32;
                       dtype="auto" would need ~35 GB). score is always null (Molmo2 has no
                       calibrated confidence; rank = emission order); k caps the returned points only
                       (no documented request-time k control). --fake mode.
    molmo_points.py    pure parser (no torch): official model-card grammar `<points coords="id x y ...">`
                       (0-1000 normalized -> pixels; 3-4 digit x/y per the official POINTS_REGEX;
                       bare `x y` pairs tolerated; out-of-image points dropped like the official
                       filter). VALIDATE against real Molmo2 output on the A100 before a study run.
  medgemma/
    llama_cpp_build.sh builds llama.cpp CUDA (sm_80) at pinned commit
                       1b29621bacc6a446bd523fc790316a24a06e886f (master 2026-10-19; treat as frozen
                       after the MedGemma smoke test, record the binary sha256).
    start_llama_server.sh  llama-server with unsloth/medgemma-1.5-4b-it-GGUF Q4_K_M (2.49 GB)
                       + mmproj-F16.gguf (0.85 GB, same repo — never mix mmproj files), --jinja,
                       ctx 8192, loopback. Port 8091. Community conversion, not Google's.
    start_bf16_server.py   OpenAI-compatible server over the official google/medgemma-1.5-4b-it
                       (AutoModelForImageTextToText, BF16, do_sample=False at temperature 0).
                       Port 8094. --fake mode (deterministic JSON replies, no torch).
    start_bf16_vllm.sh vLLM alternative for the BF16 arm (validate, not the reference).
    medgemma_client.py thin stdlib client: OpenAI chat format, base64 images, temperature 0;
                       chat_json = strict JSON parse with the explicit `uncertain` option; raw
                       reply always kept; parse failures returned (never retried).
  setup_a100.sh         one script: uv venvs -> HF downloads (GUI-Actor, Molmo2 public;
                       google/medgemma-1.5-4b-it gated: ACCEPT HAI-DEF TERMS + HF_TOKEN) ->
                       llama.cpp build -> GUI-Actor repo checkout (pinned commit
                       d98d1bbd01862f9112114b83b032f492c365a173) -> tmux services.
  smoke_test.py         one MRI slice from the design prepare dir through every service + a
                       synthetic UI screenshot sanity check for GUI-Actor (its home domain).
  tests/                pytest, --fake services via fastapi TestClient (no GPU, no torch):
                       API contracts, normalized<->pixel conversion, Molmo2 parsing incl.
                       multiple points, GUI-Actor region->point, client JSON contract.

harness/study/           (Study A/B live next to the Phase 2 study code)
  local_point_study.py   Study A runner (below)
  score_local_point.py   Study A scorer (below)
  build_decision_sets.py Study B set builder (below)
  replay_local.py        Study B replay -> MedGemma arms
  replay_decisions.ts    Study B replay -> frontier models (pi SDK, replay_candidates.ts pattern)
  score_decision.py      Study B scorer (below)
  prompts/local_point_*.md, prompts/decision_*.md   initial prompt templates (up to 8 variants
                       each will be tuned on the design split only, then frozen)

harness/study/tests/     pytest for the study code: snap-to-mark, scorer math through the
                       REUSED Phase 2 GT-hit code, McNemar/bootstrap math, resumability,
                       decision-set label logic on a tiny synthetic run tree (packbits
                       versions.npz + synthetic trajectories).
```

Validation (no GPU): `uv run --project harness/server python -m pytest harness/local_models/tests harness/study/tests -q`
(90 tests at build time). Services in `--fake` mode answer deterministically without importing torch.

## 2. A100 setup (one-time)

```bash
export HF_TOKEN=hf_...        # read access; ACCEPT the HAI-DEF terms first:
                              # https://developers.google.com/health-ai-developer-foundations/terms
cd /path/to/agseg-p5          # a full checkout WITH data/ (DVC pull)
bash harness/local_models/setup_a100.sh            # venvs + weights + llama.cpp + tmux services
python harness/local_models/smoke_test.py \
    --prepare-dir results/phase2_dev/prepare       # one MRI slice + UI screenshot sanity
```

- Downloads: `microsoft/GUI-Actor-2B-Qwen2-VL`, `allenai/Molmo2-8B` (public),
  `google/medgemma-1.5-4b-it` (gated), `unsloth/medgemma-1.5-4b-it-GGUF` (Q4_K_M + mmproj-F16;
  same HAI-DEF terms). Default weight dir `$HOME/models` (MODELS_DIR).
- llama.cpp: build script uses `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80` (A100 = sm_80).
- tmux sessions (like segd/nni): `pointd-gui` :8092, `pointd-molmo` :8093, `medgemma-q4` :8091,
  `medgemma-bf16` :8094. Loopback only. `--fake` variants exist for every service we own —
  for plumbing checks without weights, `start_bf16_server.py --fake --port 8091` stands in
  for llama-server (same OpenAI surface).

## 3. Ports + VRAM plan (A100 80 GB)

| port | service | precision | est. VRAM | notes |
|---|---|---|---|---|
| 8091 | llama-server, `unsloth/...-Q4_K_M.gguf` + `mmproj-F16.gguf` | Q4_K_M / F16 mmproj | ~5 GB at 8k ctx | deployment-realism arm |
| 8092 | pointd-gui-actor (GUI-Actor-2B + pointer head) | BF16 | ~7 GB | official repo code; verifier (+~4 GB) loads too |
| 8093 | pointd-molmo2 (Molmo2-8B) | BF16 | ~19 GB | explicit bfloat16; sdpa |
| 8094 | medgemma BF16 server (Transformers) | BF16 | ~10 GB | full-precision reference arm |
| —    | nni (nnInteractive, untouched) | — | ~5 GB | keep running from the main tree |

**Actual hardware (2026-10-20): A100 40 GB, not 80 GB** — measured idle VRAM: nni 1.0, Q4 4.3, BF16 8.9,
GUI-Actor+verifier 10.0, Molmo2 BF16 20.6 GB (sum 44.8 GB). Services run in subsets (never GUI-Actor and Molmo2
together); Study C = one pointer + one MedGemma + nni (Molmo2 BF16 + MedGemma BF16 + nni ≈ 35 GB fits, so no 8-bit
arm is needed). See `research/experiment_log_p45.md` E30. The 80 GB text below is the original plan.

All five coexist within 80 GB (~46 GB + activations/KV + CUDA context). Sequencing note from
the methodology: Study A needs no nnInteractive; Study C (later, separate task) runs one local
model at a time next to it. Measure actual `nvidia-smi` at first startup and record it.

## 4. Study A — standalone localization (RQ1, RQ4)

Inputs = the Phase 2 prepare manifests (identical inputs to the frontier study):
design `results/phase2_dev/prepare`, test `results/phase2_test/prepare`. On this worktree the
data/manifests come from the main tree (read-only): point `--prepare-dir` there. On the A100
checkout they are native.

```bash
# runner (no GT access; images + prompts only):
uv run --project harness/server python harness/study/local_point_study.py \
    --run studyA-design-gui --models gui_actor \
    --prepare-dir results/phase2_dev/prepare --reps 1
# conditions: gui_actor/molmo2 -> raw, marks; medgemma_* -> raw_box, marks_box, marks_id
# all four models in one run: omit --models (runs 4 models x their conditions x 30 cases)

# scorer (GT offline; reuses harness/study/score.py GT-hit code; aggregates fold rate
#   + a same-pass fold-off sensitivity hit@1/hit@3 and reports service latency):
uv run --project harness/server python harness/study/score_local_point.py \
    --run studyA-design-gui \
    --baseline-dir results/phase2_dev/dev-02-high --baseline-condition S
```

- Conditions: **raw** = clean image (`image_R`, pixel rulers) -> point answered in WHOLE-PNG
  pixels (the scorer subtracts the ruler margins — the panel is pasted at (40, 22) with a 26 px
  bottom margin — before `panel_to_voxel`; margin points are `outside_panel` misses); **marks** =
  marked image (`image_S`, clean left + numbered marks right) -> point snapped to the nearest
  drawn mark (predeclared rule: Euclidean over the manifest `combined_px`, ties -> lowest id);
  for MedGemma also **marks_id** (answer a mark id; non-integer ids are parse failures, unknown
  integer ids `invalid_mark`) and **marks_box** (box `[y0,x0,y1,x1]` -> centre -> snapped). The
  unsnapped raw point is scored separately as a secondary diagnostic and never mixed with the
  snapped answer.
- Prompt overrides: `--prompt COND=PATH` (repeatable) replaces the default
  `local_point_<COND>.md` template for one condition — per-model×condition prompt variants
  without touching the shared defaults. The effective path + sha256 per condition are recorded
  in meta.json `prompts` (keyed `<cond>`) and are immutable on resume; runs from before the
  rename (meta keys `local_point_<cond>`) still resume when the templates are identical.
- Clean-panel fold (declared scorer rule, `--clean-panel-fold on` default): the marks image is
  clean panel 0 + 16 px gap + marked panel 1 and both panels share the same view geometry, so a
  `marks`/`marks_box` point whose x lies on the clean panel is translated (x + panel_w + gap)
  BEFORE snapping; gap points are not folded. Per-rank `folded` records it; the aggregate adds
  `n_folded`/`fold_rate` (rank 0) and every response is also scored with the fold off in the
  same pass (`hit1_nofold`/`hit3_nofold` in score.json + report) so the fold's effect is
  measured, not assumed. Pointd `marks` ranks are deduplicated by snapped mark id (dropped
  duplicates do not consume a rank) so hit@3 means 3 distinct marks; the dedup is tied to the
  same flag and has NO effect on hit@1/hit@3 at the default k=3 (only k > 3).
- MedGemma box scale + unwrap (declared scorer rule): boxes `[y0,x0,y1,x1]`; all four values in
  [0,1] -> fractions (`unit`); else only all-integer values on [0,1000] with max > 1 ->
  `per1000` (divided by 1000); anything else — bool values, mixed/slightly-overshooting scales
  like [0.1, 0.2, 1.02, 0.4] (not reinterpreted: that would shrink them to a tiny box near
  (0,0)), a value outside [0,1000], or a degenerate box — is a parse failure. A box emitted as
  a single nested list `[[y0,x0,y1,x1]]` is unwrapped to the inner four numbers
  (`box_unwrapped: true`).
- Latency semantics: response rows carry `service_latency_ms` (SERVER-reported inference time,
  queue wait behind the per-process inference lock excluded: pointd `latency_ms` field, the
  BF16 server's `latency_ms` extension, llama-server `timings.prompt_ms + predicted_ms`) and
  `wall_ms` (client wall clock, queue wait included). The scorer's latency mean/median report
  the service latency (legacy rows fall back to `latency_ms` — chat rows there are client wall
  time, not comparable across models; the report says so).
- Service-version resume check: /healthz is fetched once per selected model at run start;
  resuming refuses when an existing ok row's `service_version` differs from the live version
  (servers 1.0.1 serialise inference; responses collected before that lock under
  concurrency > 1 may be corrupted — re-run them). llama-server exposes no /healthz version:
  its rows keep `service_version: null` and the check is skipped for that model.
- Prompt templates: `harness/study/prompts/local_point_*.md` (initial versions; {{target}} =
  histotype text, {{sequence_text}}). Up to 8 variants per model × condition — tune on the
  design split ONLY, then freeze (sha256 in meta.json) before the test split. Per-row and meta
  prompt hashes are over the raw template BYTES (`read_bytes`).
- Scorer: hit@1, hit@3 (any of ranks 0-2), EDT distance in mm, parse/abstain rates, latency,
  per-histotype breakdown, case-bootstrap CIs, and paired exact McNemar vs the Phase 2 frontier
  run (both arms use the SAME per-case summary: majority over reps of per-rep hit flags, ties ->
  miss; error/parse reps count as misses) (default baseline: the main tree's
  `results/phase2_test/p2-test-high` for test-split runs; pass `--baseline-dir
  results/phase2_dev/dev-02-high` for design tuning).
- Outputs: `results/phase5/studyA/<run>/{responses,scored}.jsonl`, `score.json`, `report.md`,
  `meta.json` (model ids, prompt hashes, git state, urls). Resumable: rerun the same `--run`
  to retry only failed calls (deterministic decoding -> reps=1 default).

Decision rule (preregistered in the methodology): a specialist wins RQ1 if hit@1 ≥ the frontier
marks baseline with McNemar p < 0.05; hit@3 ≥ 0.90 makes it useful as a *proposal* tool even
if hit@1 is lower.

## 5. Study B — decision sets (RQ2, RQ3 component level)

```bash
# build the three sets from the main tree's design runs (read-only):
uv run --project harness/server python harness/study/build_decision_sets.py \
    --runs-root /home/ubuntu/agentic-segmentation/results --out results/phase5/studyB
# built here (validated): candidate_judgement 100 rows (18 lesion) — the E16 set, expanded
#   per candidate; done_check 132 rows (10 not_done); wrong_structure 140 rows (21 wrong).
#   (phase1-08-max contributes 12 done / 20 wrong states: in that run 18 cases never made a
#   first review grid / a successful first point — the states do not exist, not a bug.)

# replay to MedGemma (each set x each arm; strict JSON with `uncertain`):
uv run --project harness/server python harness/study/replay_local.py \
    --set done_check --run medgemma-q4 --model medgemma_q4

# replay to frontier models (identical manifests + templates; minimal neutral system prompt —
# MedGemma has no system instruction, and on Anthropic OAuth accounts the provider request
# carries the fixed Claude Code identity prefix, accepted exactly and hashed per row in
# replay_decisions.ts):
npx tsx harness/study/replay_decisions.ts --set done_check --run sonnet-high \
    --images-root /home/ubuntu/agentic-segmentation

# scorer (sens/spec/AUROC + sens@spec>=0.95 threshold + bootstrap CIs + paired McNemar):
uv run --project harness/server python harness/study/score_decision.py \
    --set done_check --run medgemma-q4 --compare-run sonnet-high
```

- Sets (manifests in `results/phase5/studyB/<set>/manifest.jsonl`; labels are GT-derived and
  NEVER enter a prompt; `set_meta.json` records the rules and source runs):
  1. **candidate_judgement** — E16 manifest expanded per candidate; label = candidate precision
     > 0.5 -> lesion. Positive class lesion.
  2. **done_check** — first finish review grid per case per design run
     (phase1-06-high, phase1-07-xhigh, phase1-08-max, phase1-09-xhigh-ex, phase1-10-high-cand);
     label not_done iff a later/best version (versions.npz unpacked with `eval/v2/score.py
     unpack_version`) or the oracle-accept of GT-positive candidates (deterministic intensity
     candidates, recomputed + matched as in `prepare_candidate_replay.py`) improves Dice by
     >= 0.05. Positive class not_done (the risky miss is calling a fixable state done).
  3. **wrong_structure** — state after the first model prompt (first successful
     version-creating op, eval/v2 convention); label wrong_structure iff Dice < 0.1 (an empty
     successful version records Dice 0.0 and takes the same rule).
     Positive class wrong_structure; report sens at specificity >= 0.95.
- Answer contracts (prompts/decision_*.md): `{"answer": lesion|not_lesion|uncertain,
  "confidence": 0-1}` (candidate), `{done|not_done|uncertain}`, `{wrong_structure|structures_ok|
  uncertain}`. AUROC from `confidence` in the stated answer (p(positive) = confidence for
  positive answers, 1-conf otherwise; missing confidence -> 0.5). Parse failures are counted
  errors, never retried; uncertain rows are abstains (excluded from sens/spec, counted, and an
  uncertain-as-negative AUROC is also reported).
- Scoring/stats (score_decision.py): CIs are CASE-LEVEL CLUSTER bootstraps (resample case ids,
  carrying every decision of the case together — candidates/reps within a case are correlated);
  the paired McNemar (`--compare-run`) pairs per-decision CORRECTNESS (tp + tn, both classes;
  parse/transport failures are wrong; keys aggregated over reps by majority) so negative-class
  discordances count.
- Output per run: `results/phase5/studyB/<set>/<run>/{responses,scored}.jsonl`, `score.json`,
  `report.md`, `meta.json`.

## 6. Test-split discipline

Tune prompts/thresholds on the design split only; freeze (git tag + preregistration in
`research/experiment_log.md`) BEFORE any test-split run; the test split is 169 cases
(`--prepare-dir results/phase2_test/prepare`, baseline `results/phase2_test/p2-test-high`).
The freeze entry must restate the Study A scorer rules from §4 (clean-panel fold + dedup and
the fold-off sensitivity, MedGemma box scale + nested unwrap, latency semantics,
service-version resume check) so the frontier comparison is not open to a post-hoc-scoring
objection.

## 7. What remains (not part of this build)

- **segd v3 integration is a separate later task**: the `seg_locate` tool (Study A winner
  proposes ranked marks), the MedGemma second-reader checkpoints (first sheets / first finish /
  per candidate) injected into tool results, the v3 flags and ablation arms, and the frozen
  test-split v3 run.
- Molmo2 8-bit (BitsAndBytes) variant for in-harness use — BF16 fits standalone on 80 GB; the
  8-bit arm matters only if it must share VRAM with nnInteractive (measure first).
- GUI-Actor verifier arm as its own ablation (default off).
- Q4-vs-BF16 MedGemma quantization measurement on the design set (the two arms exist for it).
- llama-server log capture / nvidia-smi sampling around runs for the cost/latency table.
- Prompt-variant tuning (≤ 8 per model×condition) on design, then freezing.

## 8. House rules respected by this build

- The main tree `/home/ubuntu/agentic-segmentation` was touched READ-ONLY (manifests, design
  runs, E16 manifest, data). Its services/tmux sessions untouched; no GPU used anywhere in
  this build (all tests run on CPU with --fake backends).
- Nothing committed; new files only (plus a `data` symlink in the worktree pointing at the
  main tree's data dir for read-only validation — on the A100 checkout, replace it with a real
  DVC-pulled `data/`).
