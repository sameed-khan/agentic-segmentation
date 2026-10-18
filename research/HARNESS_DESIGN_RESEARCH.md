# Code-First Harnesses for Agentic 3D Segmentation

**Design memo: architecture, action-space comparison, MVP, and ablation plan**

Status: research synthesis + concrete proposal. Written for the RSNA/MSTT agentic-segmentation project.
Date of research: 2026. All citations verified by fetching the primary source unless marked otherwise.

---

## 0. TL;DR

1. **Build the harness around code, not clicks.** Treat the volume as an object in a REPL (RLM-style context offloading) and the mask as a versioned, programmable data structure. Every action is a line of Python calling a typed prelude; every observation is a number, a diff statistic, or an *artifact id* for a rendered image — never a raw pixel coordinate in the main context.
2. **Keep a small typed control surface (JSON/function-call) for admission control, budgets, and stopping**, and make code the expressive surface for the work in between. This is the hybrid: *typed gate, code interior*. Pure GUI/click action spaces are measurably hostile to current models in exactly the regime we care about (point-precise spatial control), and pure JSON tool calls cap composition and force per-click round-trips.
3. **The prototype already contains the evidence for this**: `segctl` has to repair image-derived clicks with a `--snap N` brightest-voxel heuristic because *"your pixel-coordinate estimate is off by 5–20 px"* (see `eval/skill/archive/USAGE_GUIDE_02_guide_edit.md`). That heuristic is a patch for a bad action space. Code actions with real coordinates and programmatic snapping, morphology, and connected-component logic remove the entire class of error.
4. **Decomposition should be spatial and structural, not textual**: ROI crops that re-enter the interactive model on a sub-grid, resolution pyramids (coarse seed → fine boundary band), structure-by-structure sub-tasks on multi-label volumes, and slice-holdout verification. This is the volume analogue of RLM recursion, and it is the axis most likely to produce the "OOD + decomposition" result the project wants.
5. **Assume the agent will game any proxy you give it.** Because the harness executes arbitrary code with read access to the filesystem, the leakage surface is larger than in a CLI: ground-truth files, filenames, sibling-case masks, and even the eval manifest are all reachable unless explicitly walled off. Budget real effort for sandboxing, decoy canaries, and evidence-based scoring (MedFlowBench-style) from day one.

---

## 1. The RLM line of work, verified

### 1.1 Recursive Language Models

- **Paper:** Zhang, Kraska, Khattab, *Recursive Language Models*, arXiv:2512.24601 (<https://arxiv.org/abs/2512.24601>); first posted 2025-12-31, updated 2026-05-11, listed as NeurIPS 2026. Code: <https://github.com/alexzhang13/rlm> (and a minimal reference implementation `rlm-minimal`).
- **Claim:** an RLM is a thin wrapper over an LM that puts the long *context* into a Python REPL as a **variable** and lets the root LM write code to peek, partition, transform, and map over it, launching recursive sub-LM calls as *functions in code*, then returning a final answer via `FINAL(...)`/`FINAL_VAR(...)`.
- **Reported results:** inputs up to two orders of magnitude beyond context windows; on GPT-5, "a median across the evaluated benchmarks of 26% [better] against compaction, 130% against CodeAct with sub-calls, and 13% against Claude Code" across four long-context tasks, at comparable cost; post-trained `RLM-Qwen3-8B` beats its base by 28.3% on average and approaches vanilla GPT-5 on three long-context tasks.
- **Design choices that matter for us (from the blog, <https://alexzhang13.github.io/blog/2025/rlm/>):**
  1. context = an object to be *understood*, not a prefix to be attended over;
  2. the recursive unit is invoked *inside* the REPL, so intermediate results live in variables instead of the root context;
  3. the root context stays small, so it does not rot;
  4. "no single language model call should require handling a huge context";
  5. the environment is explicitly **not** required to be a REPL — the environment is the degree of freedom.
- **Repository note that is directly on point:** the authors describe the RLM as "a bet on future 'language model' design choices… a CodeAct-style harness with sub-(R)LM calls as functions in code, and context/prompts as objects in code. We want to move away from the JSON tool-calling standard."

### 1.2 Harnesses as compositional generalizers (the most directly relevant follow-on)

- Zhang & Khattab, *Language model harnesses are compositional generalizers*, July 2026, <https://alexzhang13.github.io/blog/2026/harness/>.
- Definitions worth stealing: a harness is the program between the environment and the network that "decides how to encode the current state of the environment … into one or more inputs to the LLM and how to determine the next action." A good harness makes each individual LM call **locally in-distribution (LID)** even when the whole task trajectory is OOD.
- Mechanism: the harness induces an **equivalence relation over task states**, so structurally similar tasks produce near-identical root-context trajectories. Two ingredients: (1) **context offloading** (task-specific content lives in variables), and (2) **programmatic sub-agent calling** (tool and sub-call outputs can be consumed in code without ever entering the root context).
- Empirical claims: training Qwen3-30B-A3B-Instruct-2507 as an RLM only on **short** task splits generalizes to splits **8–32× longer**, with ~10× the eval lift at matched train lift relative to the base Transformer; and training on one domain transfers far better to other domains that share a decomposition strategy.
- **Implication for the harness we design:** the primary design objective is not "give the model a powerful tool." It is "make every individual model call see a *familiar* observation." For 3D segmentation that means: never dump an entire volume render or a long click history into context; instead hand the model a small structured summary plus handles, and let sub-calls (or cheap models) look at images.

### 1.2b Two more Alex Zhang posts worth reading for framing

- **The Mismanaged Geniuses Hypothesis** (Apr 2026, <https://alexzhang13.github.io/blog/2026/mgh/>): existing frontier models are under-utilized because individual calls are badly managed; the argument that tasks we care about "can apparently almost always be decomposed rather naturally into sub-tasks that are not only far simpler but also … not far beyond the current generation of language models." This is the conceptual license for the decomposition axis in §4.7 and §6.5.
- **Language Model "Shape"** (Sep 2026, <https://alexzhang13.github.io/blog/2026/shape/>): what it means to design a language model around a harness rather than the reverse — useful for the paper's framing section.
- The RLM repository also ships a **verifiers/`prime-rl` training environment**, i.e. the harness itself is an RL environment. If the project ever wants to *train* the segmentation policy (rather than prompt it), that is the off-the-shelf path: the same REPL-as-environment design, with sub-LM calls as functions.

### 1.3 Recursive Agent Harnesses (harness recursion, code-first sub-agent spawning)

- Lumer, Sen, Paul, Subbiah (PwC), *Recursive Agent Harnesses*, arXiv:2606.13643 (<https://arxiv.org/abs/2606.13643>).
- Recursive unit is a **full harness** (filesystem, code execution, planning) rather than a tool-less model call. A parent agent writes an executable script that spawns subagent harnesses in parallel; JSON tool-call spawning is reserved for 1–5 entries; depth-bounded.
- Controlled result with the backbone held fixed (GPT-5): Codex coding-agent baseline 71.75% → **81.36%** on Oolong-Synthetic (199 samples, context up to 4M tokens); with Claude Sonnet 4.5 as backbone, 89.77%. Stated interpretation: the gain is attributable to the harness, not the model.
- Transferable design rule: **choose the recursion channel by fan-out**. Code-execution spawning when you need many parallel units (e.g. per-slice, per-tile, per-organ, per-candidate); structured calls when the count is small.

### 1.4 Code-as-action lineage (pre-RLM, still load-bearing)

- **CodeAct** (Wang et al., ICML 2024, arXiv:2402.01030): executable Python as a *unified action space* beats JSON/text action formats by up to **20% higher success rate** across 17 LLMs on API-Bank plus a curated benchmark; multi-turn code revision, self-debugging, and library reuse come for free. Cited by the RLM authors as their starting point.
- **Code as Policies** (Liang et al., arXiv:2209.07753): LLM-generated policy code expressing feedback loops and spatial-geometric reasoning over perception outputs, with **hierarchical code-gen** (recursively defining undefined functions) — the earliest clean statement of "code is how an LLM composes spatial operations."
- **ViperGPT** (Surís et al., arXiv:2303.08128) and **VISPROG** (Gupta & Kembhavi, arXiv:2211.11559): generate Python programs that call off-the-shelf vision modules (detectors, SAM-style segmenters, image ops) and execute them. Both are the direct precedent for "program-as-spatial-action over images," and both are zero-shot, interpretable, and composable.
- **Voyager** (Wang et al., arXiv:2305.16291): persistent **skill library of executable code** + iterative self-verification; the model's competence compounds because successful programs are stored and reused. This is the template for a cross-case mask-op library.

### 1.5 Decomposition literature (why recursion helps, independent of RLM)

- Least-to-most prompting (Zhou et al., arXiv:2205.10625): break hard problems into easier subproblems solved in sequence; 99% vs 16% on SCAN with 14 exemplars.
- Decomposed prompting (Khot et al., arXiv:2210.02406): delegate sub-tasks to a library of specialized prompt-functions; sub-tasks may be recursively decomposed, or replaced by symbolic functions.
- ADaPT (Prasad et al., arXiv:2311.05772): decompose **as needed**, when the executor fails; up to +28.3% success in ALFWorld and +33% on TextCraft. This is the right policy for our planner: don't decompose everything, decompose when the current mask plateaus or the verification score is low.

### 1.6 What transfers to volumes, and what does not

| RLM / harness idea | Volume analogue | Caveat |
|---|---|---|
| Context as a REPL variable | Volume + mask store as handles in the REPL | Volumes are 10⁷–10⁹ voxels; never materialize "the whole context" as tokens. Handles only. |
| Peek / grep / partition | Numeric summarization, slice statistics, ROI cropping, connected components | Perception needs image channels, not just text search; hence render artifacts + sub-model vision. |
| Recursive sub-LM call on a snippet | Sub-harness on a **sub-volume** (ROI) or a **sub-structure** (one organ, one lesion), optionally at lower resolution | Cropping and resampling introduce coordinate transforms — the dominant bug class (§7). |
| LID observations | Each call sees one plane/report; never a 200-turn click history | Renders are expensive in tokens (§4.12); enforce a render budget. |
| Harness-induced equivalence classes | A fixed case-card schema + fixed program idioms per strategy ⇒ similar cases look token-identical at the root | Requires disciplining the prompt: forbid dumping free-form history. |
| Context rot | Long click/edit trajectories degrade exactly like long chat histories | The prototype logs every interaction; do **not** replay them into context. |

---

## 2. Programs as spatial actions: drawing, painting, CAD, viewers, pixel art

The point of this section is not a catalog; it is to extract interface rules from domains where an agent's output *is* a spatial artifact produced by code.

### 2.1 3D scenes and CAD (closest analogue: geometry from code)

- **SceneCraft** (Hu et al., arXiv:2403.01248) generates up to 100-asset Blender scenes by (a) first building a **scene graph as a blueprint** encoding spatial relations, (b) compiling the graph into Python/Blender code with numerical constraints, (c) using a VLM to analyze **rendered images** and iteratively refine, and (d) **library learning**: compiling repeated script functions into a reusable library. *Lesson:* an intermediate structured representation (graph) + a compiled program + rendered-image self-critique + a growing function library is the winning quadruple. Our analogue of the scene graph is a **plan/mask DAG**; our analogue of the library is a mask-op library.
- **CADCodeVerify** (arXiv:2410.05340): because rendering a CAD script does not tell you whether it satisfies the request, the agent **generates verification questions**, renders the object, and uses the VLM's answers to iteratively repair the code. *Lesson:* verification must be *authored*, not assumed; give the agent primitives to check its own geometric claims (mirrors §4.9).
- **BlenderLLM** (arXiv:2412.14203) and **CAD-Coder** (arXiv:2505.14646, CadQuery) show the value of a **restricted, typed DSL with an executable validator**: code is only useful if the execution environment rejects malformed geometry, and geometric reward/validation is available without ground truth.
- Community: **BlenderMCP** (<https://github.com/ahujasid/blender-mcp>) — agent writes `bpy` code against a live Blender with Python execution + render-and-look. Useful evidence that code + immediate render feedback is a workable agent loop in a real 3D tool.

### 2.2 Charts (structured geometry from code, with a benchmark)

- **MatPlotAgent** (arXiv:2402.11453): query understanding → code generation with **iterative debugging** → **visual feedback** for error correction; benchmarked on MatPlotBench (100 human-verified cases) with GPT-4V scoring that correlates strongly with human scores. *Lesson:* "generate code → render → look → debug" is the canonical loop for programmatic visual artifacts; the failure modes are (i) code errors and (ii) aesthetics/coverage errors, and they need different feedback channels (traceback vs image).

### 2.3 Vector and raster editing (the "paint through code" evidence base)

- **SVGEditBench** (arXiv:2404.13710) and **SVGEditBench V2** (arXiv:2502.19453): 1,683 original/target/instruction triplets; evaluation renders both SVGs and compares rasters plus geometric/semantic similarity (i.e., **render-mediated equivalence**, not text equality). Finding: models handle simple transformations but struggle most with **precise `<path>` manipulation**. *Lesson:* low-level geometry primitives are where code-producing models are weakest — provide high-level parametric ops, snapping, and validators rather than raw vertex arithmetic. This generalizes directly to polygon paint/erase over slices.
- Pixel-art agent demos are a surprisingly good probe of "paint through code": community skills such as `Gamezxz/pixel-art-studio` (<https://github.com/Gamezxz/pixel-art-studio>) have Claude draw every pixel via Python/Pillow, **look at its own render**, critique against a checklist, and iterate; `ccc007ccc/agent-paint-pixel-skills` (<https://github.com/ccc007ccc/agent-paint-pixel-skills>) wraps a Rust CLI + a documented file format so the agent manipulates a validated artifact instead of hand-writing RGBA arrays. A small informal evaluation (Miranda, 2025, <https://ljvmiranda921.github.io/notebook/2025/07/20/draw-me-a-swordsman/>) scores tool-calling LLMs on pixel art and finds the tool loop is the difference between plausible and unusable output. *Lesson:* agents do best when the artifact is a **validated serialized object** with CLI/API operations, and worst when they must emit raw pixel arrays by hand.

### 2.4 Distilled interface rules

1. **Programs are the medium; images are the observation.** Never let the artifact of record be a screenshot.
2. **Provide a canonical, validated artifact format** (SVG/APX/CadQuery/our mask DAG) with load/save/validate/diff, so the agent can operate on structure instead of pixels.
3. **High-level ops beat low-level geometry.** Models are weakest precisely where precision matters (paths, vertices, single voxels).
4. **Render to verify, but render on demand** and keep renders out of the root context when possible.
5. **Numerical feedback is first-class.** Geometry and constraint violations can be checked without images; e.g. SceneCraft's constraints, CAD validators, chart statistics.
6. **Library learning compounds.** Persist reusable functions; this is where "decomposition generalizes across domains" becomes measurable (RLM/harness post, Voyager).
7. **Validate-by-execution is not validate-by-intent.** Execution success ≠ task success; the agent needs explicit self-check primitives (CADCodeVerify).

---

## 3. Interactive and agentic segmentation interfaces

### 3.1 Backends and their prompt semantics

- **nnInteractive** (Isensee et al., arXiv:2503.08373; code <https://github.com/MIC-DKFZ/nnInteractive>): first comprehensive 3D interactive open-set segmentation method. Prompts: **points, scribbles, 2D bounding boxes, and a lasso**; *"leveraging intuitive 2D interactions to generate full 3D segmentations."* Trained on 120+ volumetric datasets (CT/MR/PET/3D microscopy); <10 GB VRAM; integrated into napari and MITK. Reported: point prompts +10.1 Dice over the next-best (ScribblePrompt) *while 2D models got one point per slice*; boxes +14.9 over SAM2; scribbles +23.8; interactive refinement reaches >70 Dice on all datasets and beats the best 2D model by 11.2; and the **lasso gives the best guidance signal**. **Implementation-critical detail:** nnInteractive does **not resample** (§A1: "remember that we do not resample!"), so prompt coordinates must be expressed in the *original image grid*. Its session API (`add_point_interaction`, `add_bbox_interaction`, `add_scribble_interaction`, `add_lasso_interaction`, `set_do_autozoom`) is already a programmatic surface; v2 adds `interaction_bbox` sub-crops for scribbles/lassos for a large speedup. The training pipeline itself simulates refinement by *identifying error regions* and clicking there — a good default policy for `propose_next_prompt()`.
- **VISTA3D** (He et al., CVPR 2025, arXiv:2406.05285): 3D foundation model with an automatic branch (127 classes in the CVPR paper; the code release advertises 132 classes, 7 of them tumor types) plus a **point-promptable** zero-shot interactive branch.
- **SegVol** (NeurIPS 2024 Spotlight, arXiv:2311.13385): 3D volumes with **points, 3D bounding boxes, and text**, using zoom-out/zoom-in inference to beat memory limits — an explicit precedent for our ROI/zoom abstraction.
- **MONAI Label DeepEdit / DeepGrow** (arXiv:2305.10655): a click on **one displayed slice** is encoded into a volumetric input channel and the network **re-infers the whole 3D mask**. This is the canonical "one 2.5D interaction, 3D consequence" contract and explains why an agent must reason about *where in 3D* a click lands.
- **MedSAM2** (arXiv:2504.03600), **SAM-Med3D** (arXiv:2310.15161): alternative backends; SAM-family 2D (SAM 2) needs per-slice prompting and mask propagation, and nnInteractive's own benchmark shows 2D baselines losing to 3D interactive models even when handed many more prompts. Relevant because it means **backend swap is an interface variable, not just a model variable** — the harness should abstract it (see §4.4 `backend:` namespace).

### 3.2 Agents over prompts (what exists, and the gap)

- **MedSAM-Agent** (arXiv:2602.03320): reformulates interactive medical segmentation as multi-turn decision-making. Action space (JSON tools, verbatim from the paper): `add_bbox(bbox_2d)`, `add_point(point_2d, point_type ∈ {positive, negative})`, `stop_action()`. State = the history of (action, observation) pairs; observation = the updated mask **rendered as a semi-transparent green overlay**. Two-stage training: SFT on expert-curated trajectories then agentic RL with **clinical-fidelity process rewards** to promote **interaction parsimony**; evaluated on 6 modalities, 21 datasets. *Gaps for us:* 2D, fixed three-tool space, no code, no true 3D structure editing, code-free and therefore not reusable/composable.
- **RSAgent** (arXiv:2512.24023): interleaves reasoning and **multi-turn tool invocation** with visual feedback; synthesizes multi-turn trajectories, cold-start SFT + agentic RL with process rewards. Zero-shot 66.5% gIoU on ReasonSeg (over Seg-Zero-7B by 9%) and 81.5% cIoU on RefCOCOg. *Lesson:* iterative re-localization with visual feedback beats one-shot grounding, and process rewards are what make refinement efficient rather than thrashing.
- **SegAgent** (CVPR 2025, arXiv:2503.08625): the **Human-Like Mask Annotation Task** — the MLLM emits **text-based click points** step by step, mimicking a human annotator with an interactive tool, in a multi-step MDP; fine-tuned on human annotation trajectories, comparable to SOTA without architectural changes, and additionally supports mask refinement and annotation filtering. *Lesson:* "annotator trajectory" is a legitimate action space and evaluates pixel understanding; but it is 2D, needs task-specific fine-tuning, and the actions are still per-click.
- **GenCellAgent** (arXiv:2510.13896): training-free planner–executor–evaluator loop (choose tool → run → **quality-check**) with long-term memory, routing across specialist segmenters; matches or exceeds the best tool on every one of 7 microscopy benchmarks and recovers OOD organelles. *Lesson:* an explicit **evaluator role** plus memory is what makes tool use robust; also that routing/checking is where the value is, not raw prompting.

**Synthesis:** the agentic-segmentation literature has converged on the right *loop* (propose → run → look → refine → stop) but has not yet adopted the right *action space*: all of these emit points/boxes/clicks (or RL-tuned variants of them). None of them let the agent write programs over the mask; none expose mask algebra; none do ROI recursion; none version masks; none expose a GT-free score. That is the open lane.

### 3.3 Viewer-level agents and evidence audits (evaluation design)

- **MedOpenClaw / MedFlowBench** (arXiv:2603.24649): a replayable runtime where VLM agents operate **3D Slicer and QuPath** on *complete* studies and must submit *structured evidence* (key slices, coordinates, ROIs, lesion-state fields) which is then checked against withheld masks. Findings that should shape our evaluation: (a) final-answer-only scoring is **overly optimistic** — evidence-checked scoring drops substantially; (b) **"adding image-analysis tools does not by itself solve the problem. Tools help when they make a complex procedure simple and reliable, but agents still struggle when they must choose inputs, manage viewer state, and verify intermediate outputs over multiple steps."** This is the single best empirical justification for a code-based harness: *viewer state management* is exactly what code eliminates.
- **ABRA** (Agent Benchmark for Radiology Applications, <https://luab.github.io/abra/>): drives an LLM agent through a headless **OHIF** viewer against an Orthanc PACS, scoring Planning/Execution/Outcome. Useful template for trajectory-level scoring rather than final-mask-only scoring.
- **ReX-MLE** (arXiv:2512.17838; <https://proceedings.mlr.press/v315/kenia26a.html>): 20 medical-imaging competition tasks for autonomous coding agents, with logs scored on **13 challenge strategies**; documents that general coding agents are still ineffective on domain-specific medical imaging pipelines. Our work should measure strategy quality, not just Dice.
- **AI Agents That Matter** (Kapoor et al., arXiv:2407.01502): accuracy-only evaluation produces needlessly complex, costly agents; requires **cost-controlled, holdout-based** evaluation. Mandatory reading for the ablation plan in §6.

### 3.4 The mask-operation vocabulary already exists — borrow it

3D Slicer's Segment Editor is a de-facto specification of what clinicians expect from mask editing: **Threshold, Paint, Draw, Erase, Level tracing, Grow from seeds, Fill between slices, Margin, Hollow, Smoothing, Scissors (2D and 3D, draw/erase, rectangle mode), Islands, Logical operators, Mask volume, Surface cut, Watershed, Fast marching** (<https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html>). ITK-SNAP adds contour/region-competition (snake) evolution and 3D active contours. These names are free design capital: they are already in the training distribution, they are semantically meaningful, and each has a clear programmatic signature. The DSL in §4.4 is essentially "Segment Editor as typed functions over an explicit grid, plus version control and provenance."

### 3.5 Interaction-modality evidence (what actually helps humans — and therefore likely helps agents)

- **Slice-and-Conquer** (WACV 2024): planar-to-3D pipeline — 2D interactive segmentation on one slice, then guided 3D segmentation — shows that the *best* interface is a hybrid of precise planar interaction and volumetric propagation, not one or the other.
- **Smart Brush / RBF interpolation** (J. Imaging 3(4):56, 2017): interpolate sparse annotated slices; the classic "fill between slices" abstraction. Cheap, high value, and directly implementable as `interp_slices`.
- HCI comparison of **strokes vs contour** interaction for semi-automatic segmentation (J Digit Imaging 2015, DOI 10.1007/s10278-015-9839-8): interaction modality changes both process and result quality — i.e., action-space choice is an *empirical* variable, exactly as we propose to test.
- Immersive/VR 3D contouring studies exist but do not yet show a decisive accuracy advantage over well-designed 2.5D workflows; the consistent claims are about *spatial understanding and inspection* rather than automated segmentation quality. Treat "true 3D interaction" as a research axis, not a settled win. (One practical true-3D primitive that is clearly under-used in agent harnesses and cheap to implement: **curvilinear/oblique reformations along the mask's principal axis**, which make thin, low-contrast elongated structures (nerves, vessels, fascial planes) visible in a way orthogonal tri-planar viewing cannot.)

### 3.6 What helps vs what hurts (segmentation-specific)

| Helps | Hurts |
|---|---|
| Mask algebra (boolean ops, components, holes, margin, smooth) as first-class ops | Pixel-level polygon vertex arithmetic; manually specifying brush centers |
| Explicit coordinate contract (voxel vs mm vs display) with a canonical grid | Implicit "click on what you see" with render-resolution pixel coords |
| Volumetric propagation prompts (nnInteractive lasso/scribble, DeepEdit click→3D) | Slice-local edits that silently disagree with neighbouring slices |
| Quantitative mask stats (volume mL, components, bbox mm, intensity histogram) | Screenshot-only feedback; no numbers |
| GT-free verification primitives (prompt-variant agreement, slice-holdout, RCA) | Self-reported confidence with no evidence |
| Versioned masks + diffs + undo with provenance | In-place mutation with a 10-deep undo stack and no provenance |
| ROI/zoom and coarse→fine refinement (SegVol zoom in/out; AutoZoom) | Full-volume operations on 512³ arrays (latency, memory, and token noise) |
| Structure-by-structure decomposition on multi-label volumes | Treating a multi-organ task as one binary mask |
| On-demand renders with artifact ids and cached re-use | Rendering after every action "to be safe" |

---

## 4. Recommended architecture

### 4.1 Shape of the system

```
                 ┌──────────────────────── case card (JSON, typed) ───────────────────────┐
                 │ image path(s), grid descriptor, text target, seed (if any), budgets   │
                 └───────────────────────────────────────────────────────────────────────┘
 Agent (root LM) ──writes Python──►  Harness REPL (sandbox)  ──typed prelude──►  Volume/Mask store
      ▲                                    │                                        │
      │                                    ├─ backend calls (nnInteractive session, ROI sub-harness)
      │                                    ├─ mask algebra (numpy/scipy/skimage, morphology, CC)
      │                                    ├─ verification primitives (prompt-variant, holdout, RCA)
      │                                    └─ render service → artifacts on disk (ids only)
      │
      └── observations: numbers + diff stats + artifact ids (+ optional sub-model vision answers)

 Controller (outside the REPL, typed/JSON): budget accounting, op admission, destructive-edit guard,
 stop/accept decision, final mask commit, evidence bundle export.
```

Two loops: an inner **code loop** (cheap, local, deterministic, hundreds of steps) and an outer **model loop** (expensive, few turns). The design goal is to push work from the outer loop into the inner loop, exactly as RLM pushes reading into code.

### 4.2 Grids, coordinates, and invariants

Canonicalize once, then never let a bare coordinate cross a boundary without a unit tag.

- On ingest: `nib.load(path)`, record original `affine`, `shape`, `header.get_zooms()`, and axis codes; optionally `as_closest_canonical` to RAS+ but **keep the original grid as the authoritative prompt grid for the backend** because nnInteractive does not resample.
- Expose exactly three namespaces and make them explicit in every signature:
  - `vox=(i,j,k)` — integer indices into the (possibly cropped) array;
  - `mm=(x,y,z)` — RAS+ millimetres via the affine;
  - `disp=(u,v,plane,slice)` — render-relative coordinates *only* used by the human GUI adapter, never by the agent.
- Every coordinate-bearing call requires `in_="vox" | "mm"`. Bare tuples are a type error. Rationale: DICOM is LPS and NIfTI is RAS — "the only difference between the two conventions is that the sign of the first two coordinates is inverted" — which is a left/right flip in the patient (<https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html>; <https://nipy.org/nibabel/coordinate_systems.html>). Silent flips are among the most dangerous bugs in medical image pipelines.
- A `Grid` object carries the transform and is *the* thing passed to sub-harnesses:
  ```python
  g = case.grid                      # canonical grid handle
  g.crop(bbox_mm) -> Grid            # sub-grid with .to_parent / .from_parent
  g.resample(spacing_mm, order) -> Grid   # never used for prompting, only for analysis/rendering
  ```
  Every derived grid records `parent`, `mode`, and the affine composition; masks carry their grid id. Mixing masks from different grids is a hard error.
- Rendering always annotates the displayed planes with **mm world coordinates and voxel indices** of plane centres (so the model can go from "what I see" to a number), and always reports `long_edge_px` and `render_id`.
- Invariant tests to run in CI on a synthetic phantom: `to_mm(to_vox(p)) == p`; a known-right structure renders on the correct side after every supported reorientation; cropped-grid masks paste back within 1 voxel; `resample` round-trip preserves Dice ≥ 0.99.

### 4.3 Volume and mask store (versioned DAG)

Masks are immutable nodes; operations return *new* handles. There is no in-place edit.

```python
m0 = masks.empty(g)                       # m0
m1 = seg.point(m0, vox=(120,140,75), label=+1, backend="nnint")   # model-driven
m2 = edits.close(m1, r_mm=2.0)
m3 = edits.keep_largest(m2, connectivity=26)
masks.diff(m1, m3)   # {added: 18422, removed: 903, dice_between: 0.94, bbox_delta_mm: [...]}
masks.lineage(m3)    # [m0 → m1 → m2 → m3] with op, args (in mm), code hash, timestamps
masks.rollback(m1)   # new handle equal to m1; the DAG is never destroyed
masks.finish(m3)     # declares the accepted node
```

- Node key: `hash(grid_id, parent_hash, canonical_op_args, code_text_hash, rng_seed)` ⇒ content-addressed, dedupes identical work, makes caching trivial.
- Every node stores: op name, **args in physical units** (mm, mL, voxel radii), plus the *code text* that produced it. The code text is essential for failure attribution ("which program, which line, produced the over-growth?").
- Provenance enables the two most valuable analyses in the ablation plan: **script-level failure attribution** and **oracle-op counterfactuals** (§6.7).
- Storage: `.npy` (or zarr for big volumes) arrays + a JSON DAG; large volumes memory-mapped so the REPL never OOMs.
- **Destructive-edit guard** (see §4.6): ops whose diff exceeds thresholds require `allow_destructive=True` and automatically emit a before/after render artifact and a numeric diff into the observation.

### 4.4 The typed prelude (the API/DSL)

Python is the language; the prelude is the domain layer. Namespaces:

```python
# ---- case & grid -----------------------------------------------------------------
case                    # .grid, .image, .modality, .text, .case_id, .budgets_left
g = case.grid
world(vox) / voxel(mm) / plane_axis("axial"|"coronal"|"sagittal") / axis_of(view) / slice_count(axis)

# ---- interactive backend (typed pass-through) ------------------------------------
seg.point(m, vox=..., label=+1, backend="nnint", snap="none"|"bright"|"dark", snap_radius_vox=3)
seg.box(m, lo_mm=..., hi_mm=..., backend="nnint")
seg.scribble(m, plane, polyline_vox=[[...],[...]], label=-1)
seg.lasso(m, plane, polygon_vox=[...], label=+1)
seg.reset(m)                       # clears interaction history for the session (kept in DAG)
seg.probs(m) -> ProbVolume         # soft predictions if the backend exposes them (see §4.5)

# ---- edits (Segment-Editor-class vocabulary) -------------------------------------
edits.threshold(m, lo_hi_percentile=(5,95), restrict_to=m, out="intersect"|"replace")
edits.grow_from_seeds(m, seeds=..., lower=..., upper=..., neighborhood=26)
edits.region_grow(m, seed_vox, tolerance_sigma=2.0)
edits.morph(m, op="open"|"close"|"dilate"|"erode", r_mm=2.0)
edits.keep_largest(m, connectivity=26) / edits.remove_small(m, min_ml=0.5)
edits.fill_holes(m, close_plane="axial"|"coronal"|"sagittal"|"3d")
edits.hollow(m, thickness_mm=2) / edits.margin(m, r_mm=2, sign=+1|-1)
edits.smooth(m, sigma_mm=1.0, method="gaussian"|"median"|"joint")   # joint = image-guided
edits.interp_slices(m, axis, stride=4)          # fill-between-slices / RBF
edits.boolean(a, b, op="union"|"intersect"|"subtract"|"xor")
edits.hull(m, kind="3d"|"per_slice")            # convex hull constraint
edits.paint_disk(m, plane, center_vox, r_vox, label)
edits.erase_region(m, bbox_mm)                  # "scissors" as an explicit bbox/plane cut
edits.axis_band(m, axis, lo_idx, hi_idx, keep=True)   # cheap slab surgery
edits.limit_to_intensity(m, lo_pct, hi_pct)     # explicit, logged

# ---- numeric perception (the default observation channel) ------------------------
stats(m) -> {vox, ml, bbox_vox_mm, centroid_vox_mm, n_components, comp_sizes_ml,
             surface_mm2, sphericity, per_axis_extent_mm, slices_with_mask[axis],
             intensity: {in: {p5,p50,p95}, out_shell: {...}, boundary_contrast},
             grid_id, node_hash}
profile(m, p0_mm, p1_mm, n=64) -> intensities            # boundary probes
top_slices(m, axis, n=5, by="area"|"change") -> [(idx, area_mm2, centroid_vox)]
propose(m, kind="false_negative"|"uncertain"|"boundary"|"missed_component", k=5)
    -> [ {vox, mm, why, score} ]        # from probs() when available, else distance/intensity heuristics
disagree(a, b) -> DisagreementVolume    # for prompt-variant / holdout verification

# ---- render artifacts (expensive; ids only in context) ---------------------------
render(m, view="ortho", center_mm=..., zoom="auto"|"roi"|"mm:120", max_side=768,
       overlay="outline"|"fill", extras=["prompts","prob_band","diff_from_parent","cpr"])
    -> {render_id, path, long_edge_px, planes: [{axis, idx, center_mm}], cached: bool}
render.mip(m, axis, slab_mm=(lo,hi), max_side=640)
render.cpr(m, axis="skeleton")          # curvilinear reformation along the mask principal axis
render.diff(a, b, view="ortho") -> render_id
ask(render_id, question, model="small") -> str      # programmatic sub-call: image stays out of root ctx

# ---- verification (GT-free) -------------------------------------------------------
verify.prompt_variant(m, variants=[("axial_seed",...), ("coronal_seed",...), ("tight_box",...)])
    -> {dice_between: 0.91, disagree_ml: 2.3, worst_region_vox: ...}
verify.slice_holdout(m, axis, hold_idx)   -> {agreement: 0.94, leaked_slices: [...]}
verify.rca(m, atlas=None)                 -> {rca_score, verdict:"plausible"|"suspicious"}
verify.registration_sanity(...)           # optional: QIBA-style harmonization checks

# ---- memory & flow ---------------------------------------------------------------
notes.append(fact, evidence=[render_id|node_hash], ttl="case")
plan.set(tasks=[...]); plan.status()
lib.save(name, fn, tests)                 # library learning (cross-case; reviewed before use)
budget() -> {turns, tokens_in, tokens_out, renders, seconds, dollars, interactions}
finish(m, confidence="high"|"medium"|"low", rationale=str, evidence=[...], 
       abstain_reason=None)
```

Design rules for the prelude:

1. **Every function is total and logged.** No silent failures; each returns a structured object and appends to the trajectory with its arguments resolved to physical units.
2. **`ml`/`mm` are the reporting units**, voxels only near the backend boundary (nnInteractive's native space is voxels).
3. **`snap` exists but is explicit and bounded.** Because the backend needs a voxel, a `snap_radius_vox` with a reported displacement is fine; silently snapping without reporting is not.
4. **Costly ops are declared** (`render.*`, `verify.prompt_variant`, `seg.*` re-inference) so the controller can meter them.
5. **Do not expose `subprocess`, `open`, `os`, network, or PIL** in the sandbox; the only way to write is via `finish`/`lib.save` and the render service.

### 4.5 Perception design: numeric-first, artifact-second, sub-model third

Three channels, in increasing cost:

**Channel 1 — numeric summaries (default).** The most valuable, most under-used observation in existing harnesses. From `stats()` the agent gets extent in mm per axis, per-axis slice counts, number of components and their volumes, sphericity, intensity percentiles inside vs in a 2-voxel outside shell, and a `boundary_contrast` scalar. Enough to catch: wrong component, missed satellite, gross over/under-growth, truncation at the volume edge, and masks whose boundary sits in flat noise (low contrast ⇒ suspect).

**Channel 2 — rendered artifacts on demand.** Rendered image bytes go to disk; the context receives `{render_id, path, long_edge_px, planes:[...], cached}` plus a one-line summary. Choose defaults that maximize information per image: a 3-panel ortho montage through a specified `center_mm` (not a fixed slice index), with outline overlay, a mm scale bar, plane annotations, and options for `prob_band`, `prompts`, `diff_from_parent`. For thin/elongated structures add `view="cpr"` (curvilinear reformation along the mask principal axis) and `mip` slabs — the two views in which low-contrast 3D structures become perceptible.

**Channel 3 — programmatic vision sub-calls.** `ask(render_id, question, model="small")` performs a *sub-LM call with the image attached* and returns a short string. This is the direct analogue of RLM's `llm_query` inside the REPL: the image never enters the root context, and the root stays locally in-distribution. Typical uses: "does the outline reach the posterior edge of the lesion on this slice? yes/no/uncertain", "is there a separate bright nodule ~2 cm right of the mask?", "which of the 9 slices in this contact sheet shows the largest hollow area?" Cost: one small-model vision call ≈ hundreds of tokens instead of full-resolution images in the main history, and it keeps root-context token mass flat across a long trajectory.

**Uncertainty consumption.** If the backend exposes soft predictions (`seg.probs`), derive `propose(kind="false_negative")` from the *false-negative* region rather than from the mask boundary; nnInteractive's own training loop uses exactly this error-region-clicking policy, so it matches the model's inductive bias. If soft predictions are unavailable, approximate with (a) distance transform of the current mask, (b) intensity outliers in the outside shell, (c) `disagree()` between prompt variants.

### 4.6 Execution sandbox and safety

Assume the code will do something you did not intend.

- **Process isolation:** the REPL runs in a child process (or container: nsjail/firejail/Docker; RLM's own library supports `local`, `ipython`, `docker`, `modal`, `prime`, `daytona`, `e2b`) with CPU-seconds, RSS, wall-clock, file-descriptor, and array-size caps; no network namespace; read-only mounts for image data; a single writable scratch dir.
- **Filesystem policy:** the only readable roots are the case's image file(s) and the harness's own docs. Ground truth, manifests, sibling-case directories, and eval scripts are *not mounted*. Add **canary/decoy** files (e.g. `gt_decoy.nii.gz` with wrong content, and a decoy `manifest.csv` with implausible Dice numbers) plus a monitor that flags any `open()` attempt on them — this converts leakage from a silent bias into a measurable event (see §7).
- **Determinism:** seeded RNG passed into the sandbox; log the seed; identical code + seed + backend version must give identical masks (mask hashing makes this checkable nightly).
- **Destructive-op guard:** if an op would remove >30% of current foreground voxels or multiply volume by >3×, the harness requires `allow_destructive=True`, auto-renders the diff, and (by default) keeps the parent alive as a rollback candidate. Hard rule: **no operation may silently replace the accepted mask**; only `finish()` sets the accepted node.
- **No source mutation:** the input image and (if present) the provided seed are immutable; the harness writes only to the case's output directory, with a final schema check (dtype, geometry, affine equality with input, values ⊂ {0,1} or label set).
- **Audit:** a machine-readable JSONL log of every call (op, args in physical units, code hash, node hash, wall time, tokens, render ids) plus the sandbox's file-access log. This log *is* the paper's supplementary material.

### 4.7 Hierarchical planning and recursion

Four decomposition axes, chosen by the controller based on the case card and observed state:

1. **Spatial ROI recursion** (the RLM analogue): crop a sub-grid around the mask bbox with a margin, then run `seg.*` on the crop in a *child harness instance*. Benefits: smaller arrays, faster backend (sub-crop interaction tensors are dramatically faster for scribble/lasso in nnInteractive v2), and better local resolution for small lesions. Cost: a `Grid` transform, and a paste-back step with a validity check. Rule of thumb: recurse when `mask bbox` covers <25% of the volume or when the lesion is small relative to spacing.
2. **Resolution pyramid**: coarse pass at 2–4× spacing (or a downsampled copy) to establish extent, then a fine boundary band pass (a 2–4 voxel shell around the coarse mask) for the boundary. `edits.morph` + `seg.point` on the shell only. This is the cheapest structural win for large, high-contrast lesions.
3. **Structure recursion** (multi-label / multi-structure volumes): one sub-task per label, each with its own budget, notes, mask node, and verification. On BTCV/AMOS/TotalSegmentator-scale tasks this is where the RLM-style "isomorphic decomposition" claim is actually testable (§6.5, E2).
4. **Lesion/component recursion**: `stats().n_components > 1` triggers per-component sub-tasks with explicit "is this a satellite or an artifact?" decisions, using `propose(kind="missed_component")`.

Planner shape: a small explicit `plan` object (JSON) with tasks `{id, goal, state, budget, evidence[], exit_criteria}`. Policy: **decompose as needed** (ADaPT), i.e. only when (a) verification consistency < τ, (b) the composite score has plateaued for 2 proposals, or (c) the case card says complexity > threshold (large bbox, low contrast, multi-label). Depth cap 2 (root → ROI/structure child → boundary child); fan-out cap 8 by default (see RAH: code-spawning for many units, typed calls for ≤5).

### 4.8 Proposal → evaluation → acceptance loop

Keep a small portfolio of candidate masks rather than a single evolving one:

```
candidates = [c0 (seed), c1 (after coarse), c2 (after boundary), ...]   # k ≤ 3 live
score(c) = w1*verify.prompt_variant(c).dice_between
         + w2*boundary_contrast_z(c)
         + w3*shape_prior(c; text, modality)          # e.g. sphericity/elongation plausibility
         + w4*verify.slice_holdout(c).agreement
         - w5*size_implausibility(c; text, spacing)    # gross-volume sanity
         - w6*provenance_penalty(c)                    # penalty for destructive ops w/o verification
accept new candidate iff score improves by > ε AND the change is explainable
   (the op that produced it is named in the rationale)
```

All six terms are GT-free and cheap; all six are individually hackable (see §7), which is why (a) the score is multi-term, (b) the weights and thresholds are **frozen before the main run** and tuned only on a disjoint pilot split, (c) ties are broken by *evidence quantity* (a candidate that has a verification render attached beats an unattached one), and (d) the paper reports the score's correlation with Dice (Spearman) as a first-class result — if the score does not predict Dice, the agent has nothing legitimate to optimize and the harness must not pretend otherwise.

### 4.9 Verification and confidence

Implement four complementary GT-free checks, each returning a scalar plus a locatable failure region:

1. **Prompt-variant agreement:** re-run the backend from 2–3 *different* prompt strategies (axial seed, coronal seed, tight box/lasso) and measure Dice/disagreement between the resulting masks. Disagreement localizes ambiguity; high disagreement should *suppress* confidence and direct refinement.
2. **Slice holdout:** mask the volume from `verify.slice_holdout` (exclude a slab), re-run the backend, and score agreement inside the held-out slab itself vs the predicted mask there. This catches "the mask is only right where the prompts were" — a direct, cheap test of overfitting to visible slices.
3. **Reverse classification accuracy (RCA):** predict quality by registering a propagated label set back to the case (Robinson et al., MICCAI 2018 <http://www.doc.ic.ac.uk/~bglocker/pdfs/robinson2018miccai.pdf>; UK Biobank QC, JCMR 2019 <https://link.springer.com/article/10.1186/s12968-019-0523-x>). Use it as an *independent* failure detector whose verdict is compared with the agent's own confidence (the project already reports recall 0.83 for flagging Dice<0.5; an external detector is a stronger claim).
4. **Contrast/geometry sanity:** boundary shell contrast z-score, mask touching the volume edge, component count vs expectations, volume vs a plausible size band derived from text + spacing. Cheap, and it is the only channel that reliably flags thin/low-contrast misses (the lesion is "found" but truncated).

Confidence output must be **evidence-backed**: `finish(confidence=..., evidence=[render_id, verification_ids])`; a confidence label without evidence is recorded as such and scored separately. Report calibration (ECE, AUROC of "Dice<0.5") and *selective risk at coverage* — the operationally meaningful framing if a human is going to review only the flagged cases.

### 4.10 Memory

- **Per-case notes:** bounded, schema-validated, agent-written facts each linked to evidence ids (never free-form unbounded text — that is how context rot starts). Cap by count and rewrite policy; older notes are summarized into the case card, not accumulated.
- **Case card persistence:** the plan, current accepted node, budget counters, and verification history — all outside the token context, read on demand.
- **Cross-case library (`lib`):** Voyager-style. The agent may propose a reusable function (e.g. `boundary_band_refine(mask, sigma)`, `satellite_check(mask, k)`) with a name, docstring, and unit tests on synthetic phantoms; the harness runs the tests, a second model reviews for safety (no filesystem/network, bounded loops), and only then is it committed to the library and exposed in the prelude under `lib.<name>()`. **This is the mechanism by which cross-case generalization becomes measurable**: run cases in a fixed order and test whether late cases benefit (E7 in §6.5). Note the honesty requirement: library learning *is* training on the run, so hold out a histotype split for the transfer claim.

### 4.11 Stopping

Explicit, logged, and counterfactually auditable:

```
stop if  (a) score plateau: Δscore < ε for 2 consecutive candidates, OR
         (b) verified: prompt-variant Dice_between ≥ 0.95 AND slice_holdout agreement ≥ 0.95, OR
         (c) budget exhausted (any of turns/tokens/renders/dollars/seconds), OR
         (d) two consecutive no-op or invalid actions, OR
         (e) hard blocker detected (backend failure, volume-edge truncation, ambiguity unresolved)
and always: emit confidence + evidence + stop_reason; if the best score < τ_low, emit abstain/low.
```

Because every mask node is stored, we can compute **oracle stopping** offline: `Dice(best node)` vs `Dice(final node)` ⇒ `stopping_regret`, and `Dice(final)` minus `Dice at k interactions` ⇒ interaction-efficiency curves. This converts the stopping policy into a measurable quantity with no extra runs (an idea borrowed from the interactive-segmentation NoC literature and made cheap by the mask DAG).

### 4.12 Cost and latency engineering

- **Token math is the budget.** For Claude-family models, images are tokenized as 28×28-pixel patches (`⌈w/28⌉ × ⌈h/28⌉` visual tokens; long edge capped at 1568 px for standard resolution), which the docs and community analyses summarize as ≈ `(w×h)/750` — so a 768×768 render ≈ 786 tokens, a 1024×1024 ≈ 1.4k tokens, and a 9-slice contact sheet at 1024 px ≈ 1.4k tokens *for nine slices* (the reason contact sheets are smart). Sources: Anthropic vision docs <https://platform.claude.com/docs/en/build-with-claude/vision>. Re-measure on your own model/version — the constant and caps differ by model and provider, and file size is irrelevant.
- **Default to numbers; spend tokens on montages and difference views, not on repeated full slices.**
- **Cache aggressively**: render cache keyed by `(grid, node_hash, view, overlay, max_side)`; mask-node dedupe by content hash; `propose()` results cached per node.
- **Overlap tool latency with generation.** Code-first harnesses can speculatively pre-launch predictable calls (the pattern Zhang calls *speculative programmatic tool calling*, <https://alexzhang13.github.io/blog/2026/spec-ptc/>): while the model is still emitting tokens, pre-execute calls whose arguments are already syntactically complete and non-dependent, and launch independent backend calls in parallel when the generated program does not express them as async. Concretely: run the three prompt-variant backend calls concurrently; pre-render the "standard ortho" artifact for the current node as soon as a `seg.*` call parses. Report both wall-clock and GPU-seconds.
- **Cap with accounting, not hope**: per-case caps on turns (e.g. 25), renders (e.g. 6), backend calls (e.g. 12), tokens, and dollars; the controller refuses over-budget calls with a structured error the agent can read (`error.code = BUDGET_EXCEEDED`, with the remaining budget). This is also the mechanism that makes arm comparisons fair (§6.6).
- **Backend latency is the floor**: interactive models are the slow part (nnInteractive advertises rapid inference and <10 GB VRAM; measure your own per-interaction wall time, and use `interaction_bbox` sub-crops for scribble/lasso, which the authors call out as a large speedup). Batch ROI sub-harness calls; avoid re-running the backend at full resolution when a shell-only refinement will do.

### 4.13 Interfaces

- **Python package + REPL** (primary, for the agent).
- **MCP server wrapper** (<https://modelcontextprotocol.io/specification/2025-06-18>) exposing a *typed subset* (case card, `stats`, `render`, `finish`) so other coding agents (Claude Code, Codex, Cursor) can drive it; note the existing prior art in this space — `mcp-slicer` (<https://github.com/zhaoyouj/mcp-slicer>) and `pieper/slicer-skill` (<https://github.com/pieper/slicer-skill>) both expose **`execute_python_code`** as a tool, i.e. the community's naive version of this design. Our contribution is the disciplined prelude + grids + versioning + verification, not the socket.
- **Human GUI adapter** (optional, for review and for the human-collaboration arm): the existing `segctl` viewer becomes a *client* of the same store; a click maps to `(grid, plane, mm)` deterministically, and every human edit creates a node like any other action. This gives a clean human-in-the-loop baseline without a second codebase.
- **Export**: `mask.nii.gz` + `trajectory.jsonl` + `metadata.json` (already the prototype's contract), plus DICOM-SEG (<https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html>) and RT-STRUCT (<https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.html>) for clinical handoff, and a machine-readable **evidence bundle** (key slices, coordinates, verification scores) to support MedFlowBench-style evidence scoring.

### 4.14 Controller loop (pseudocode)

```python
card = load_case_card(case_id)                 # image(s), grid, text, seed?, budgets
root_ctx = [card.short(), prelude_docs(), plan_template()]
while not stopped:
    obs = repl.execute(model_turn(root_ctx))   # model writes one or more programs
    obs = {**numbers_obs, "artifacts": artifact_ids, "errors": errors}
    repl.update_plan(card, obs)                # controller-side checks: guards, budgets
    if need_vision(obs):                       # cheap decision, not a model decision
        obs["vision_answer"] = ask(obs["artifacts"][0], fixed_question, model="small")
    root_ctx.append(compact(obs))              # compact() = < ~200 tokens, no history replay
    if stopped_by_rule(...):
        break
final = masks.best_node()
finish(final, confidence=evidence_backed_confidence(final), evidence=evidence_ids(final))
write_evidence_bundle(case_id, final, trajectory, renders, verification)
```

The key discipline: `compact(obs)` never re-inserts raw image bytes, never replays the click history, and always carries budget status. The root context of a 25-turn case should stay in the low tens of thousands of tokens.

---

## 5. Action spaces compared

### 5.1 The three candidate action spaces

| | **A. GUI / screenshots + clicks** | **B. JSON function calls with coordinates** | **C. Code / DSL programs over volumes** |
|---|---|---|---|
| Atomic unit | click/drag at display pixel | one tool call with args | one program (many ops, loops, branches) |
| Precision | display-resolution, then snapped | exact once the model emits the right number | exact, and computed by the runtime |
| Spatial grounding burden | model must map image → pixel → voxel; error 5–20 px in practice (see `segctl --snap`) | model must emit coordinates it usually cannot see precisely | model names *objects and quantities*; runtime resolves them |
| Composition (loops, conditionals, reuse) | none | none (one call per step) | native |
| Round-trips per unit of work | 1 per click | 1 per call | ~1 per program |
| Token cost | image per turn + full history | arg text per call | code text once; observations compact |
| Auditability | low (pixel coords, screen state) | medium | high (code + physical-unit args + DAG) |
| Reproducibility | poor (viewer state, layout) | medium | high (seed, hashes) |
| Error surface | grounding errors, viewer-state errors, silent misclicks | coordinate/unit errors, arg typos | logic bugs (inspectable, testable, replayable) |
| Hacking surface | low (hard to cheat with a click) | medium | **high** (arbitrary code can read files) |
| Learnability | poor | medium | good (RL-friendly; cf. RLM training harness) |
| Evidence | low | medium | high |

### 5.2 What the evidence says

- **Code > JSON/text action formats**, with the same model: CodeAct reports up to **20 point higher success** with executable code actions across 17 LLMs (arXiv:2402.01030).
- **Code + context offloading + recursion >> other long-context scaffolds** with the same model: RLM reports median +26% vs compaction, **+130% vs CodeAct with sub-calls**, +13% vs Claude Code across four long-context tasks (arXiv:2512.24601). Note the nuance: RLM *beats* plain CodeAct by adding context offloading and recursive sub-calls, not merely by using code.
- **Harness choice changes generalization**, not just efficiency: RLM training generalizes to 8–32× longer contexts with ~10× the eval lift of the base Transformer (harness post, July 2026); RAH improves a coding-agent baseline 71.75% → 81.36% with the backbone fixed (arXiv:2606.13643).
- **Pixel-precise GUI control is a catastrophic action space for current VLMs**: on PAGE Bench (point-precise geometric GUI construction), general multimodal models exceed **88% action-type accuracy while staying below 6% task success**; specialised GUI agents are below 9% step success (arXiv:2605.15963). On ScreenSpot-Pro (high-resolution professional UIs) the best model reaches **18.9%** grounding accuracy, and a cascaded *search* wrapper gets 48.1% (arXiv:2504.07981). Coordinate-free grounding methods (GUI-Actor) beat even a 72B coordinate-emitting model (40.7/44.6 vs 38.1) (arXiv:2506.03143). Set-of-Mark (arXiv:2310.11441) is the same lesson from the opposite direction: **the fix for coordinate prediction is to stop predicting coordinates** and let the model *select* runtime-provided candidates.
- **Viewer-state management is a real, measured tax**: MedOpenClaw/MedFlowBench finds that adding analysis tools does not by itself help agents, which "still struggle when they must choose inputs, manage viewer state, and verify intermediate outputs over multiple steps" (arXiv:2603.24649).
- **Segmentation agents that do emit clicks need RL to use them efficiently** (MedSAM-Agent's process reward for "interaction parsimony", arXiv:2602.03320; RSAgent, arXiv:2512.24023) — i.e. the click action space is expensive enough that people train models to spend fewer clicks. Code actions make many of those clicks unnecessary (a loop, a boolean op, a component filter).

### 5.3 Recommendation: hybrid, code-first, with a typed gate

Verdict: **C as the work surface, B as the gate and the contract, A only for humans.**

- **Everything inside a work step is code (C).** Prompting, mask algebra, ROI recursion, verification, rendering requests, and the final commit.
- **The controller exposes a small typed surface (B)** for: case bootstrap, budget status, `finish(m, confidence, evidence)`, and every *costly or irreversible* action (backend invocations, destructive ops, `lib.save`). This is not a competing action space: it is the admission-control layer that lets the harness enforce budgets, guards, and stop rules without trusting the model. It also preserves the option of grading B vs C fairly in the ablation (§6.2, arms B/D).
- **No raw coordinates in the agent's context (A).** The agent may pass *indices and mm*, but it should never be asked to produce a display-pixel coordinate. If a render must be clicked (human collaboration, or a GUI-agent ablation arm), the mapping is performed by the harness from render metadata to `(grid, plane, mm)`, exactly as the prototype's `--from-render` already does — and the harness records the snapped displacement.
- **Coordinates are addressed by handles, not literals.** `propose(m, kind="false_negative")` returns candidate points; the agent chooses one *by id or by index into a list*, mirroring Set-of-Mark/GUI-Actor's winning pattern. This is the single highest-leverage change relative to both the prototype and the JSON-tool agents in the literature: the model stops being a coordinate estimator and becomes a *selector and composer*.

Why the hybrid rather than pure code: (i) budgets and safety cannot be enforced by trusting generated code; (ii) the paper needs a clean C-vs-B comparison, which requires B to exist as a first-class arm; (iii) some calls (backend inference) are expensive enough that per-call metering is worth the round-trip; (iv) human handoff needs a typed, auditable interface. Why not pure JSON: composition, reuse, context economy, and the measured failure of coordinate-emitting action spaces.

---

## 6. MVP and staged ablation plan

### 6.1 MVP (target: 1–2 weeks of engineering before the first experiment)

Non-negotiable scope:

1. **`Grid` + ingest** with the invariant test suite (phantom, reorientation, crop/paste round-trip).
2. **Mask DAG store** (content-addressed, `.npy` + JSON, lineage, diff, rollback, `finish`).
3. **Prelude v1**: `seg.point/box`, `edits.*` (morph, keep_largest, remove_small, fill_holes, boolean, interp_slices, threshold, limit_to_intensity), `stats`, `propose`, `render`(ortho/mip/cpr), `ask`, `verify.prompt_variant` + `verify.slice_holdout`, `finish`.
4. **Sandbox + audit log + budgets + destructive guard + canaries.**
5. **Controller** with the stop rules and the composite score (weights frozen after pilot).
6. **Runner** on the existing MSTT manifest (166 cases; reuse `eval/lib/reorient.py`, `eval/lib/dice.py`, and the manifest CSVs), plus the existing one-click baseline (`eval/run_oneclick_baseline.py`) as arm F0.
7. **Trajectory tooling**: replay a logged run offline; compute Dice at every node; produce the oracle-stopping/regret report. *Build this before the main run* — it is the analysis engine for every experiment below.

Explicitly out of scope for MVP: VR/3D interaction, cross-case library learning (phase 2), DICOM-SEG export (phase 2), multi-model ensembles.

### 6.2 Arms (fair baselines)

| Arm | Action space | Feedback | Decomposition | Notes |
|---|---|---|---|---|
| **A — CLI (prototype)** | `segctl` commands with `--from-render` pixel clicks + `--snap` | renders + stats | no (single loop) | existing artifact; the incumbent |
| **B — Code, numeric-only** | DSL code | numbers/diffs only (no images) | no | isolates "code without vision" |
| **C — Code + hybrid perception** | DSL code | numbers + artifact renders + `ask` sub-calls | no | **primary candidate** |
| **D — JSON tools with coordinates** | same primitive set, one call per action, coordinates in args | renders + stats | no | isolates **code vs JSON** (the headline claim) |
| **E — C + decomposition** | DSL code | as C | ROI recursion + pyramid + structure/component sub-tasks + planner (ADaPT-style) | isolates **decomposition** |
| **F0 — one-click backend** | n/a | n/a | n/a | nnInteractive with the provided seed only (`run_oneclick_baseline.py`) |
| **F1 — published baselines** | n/a | n/a | n/a | U-Net / LiteMedSAM per the dataset paper; plus existing per-histotype published numbers |
| **F2 — oracle ceiling** | n/a | n/a | n/a | GT-derived prompt set (3 prompts, best of 3 seeds) — an upper reference, *not* a competitor; report as leakage-sensitive ceiling |
| **G — GUI/computer-use agent** | real screenshots + clicks on the viewer | screenshots | no | quantifies the A-space penalty in *this* domain; run on a 40-case subset |
| **H — E + library learning** | as E | as C | as E + `lib` | phase 2; needs a fixed case order and a held-out histotype |

Fairness controls (mandatory):

- **Identical information**: same seed (or none), same text, same image files; no modality flags beyond what the arm can infer; GT never mounted.
- **Equal budgets, three ways**: (i) tool-call budget, (ii) wall-clock budget, (iii) input/output token budget. Report results under all three, plus the *unconstrained* run, so cost-accuracy trade-offs are visible (AI Agents That Matter, arXiv:2407.01502).
- **Same backend**: pin the nnInteractive checkpoint hash and the harness commit; record both in the case metadata.
- **Same model & sampling**: one model per experiment (e.g. Sonnet 4.6), fixed temperature/top-p, fixed seed; 3 repeats on a 40-case stratified subset to estimate variance.
- **Same prompt skeleton** where the action space permits; document the unavoidable differences (a code prelude vs a CLI usage guide) and report them as a limitation, since prompt engineering is a confound by construction. Mitigation: give each arm the *best* documentation you can write for that arm, and report the documentation as an appendix.

### 6.3 Metrics

Primary endpoint (pre-registered): **median Dice on MSTT-iso, arm C vs arm A**, paired by case.

Secondary: mean Dice; IQR; Dice<0.5 rate; HD95 (mm); NSD@2mm; volume bias (signed %); Dice@1 interaction-call (parsimony); interactions/case; renders/case; wall-clock/case; tokens in/out; $/case; GPU-seconds; **stopping regret** (Dice(best node) − Dice(final node)); **interaction-efficiency AUC** (Dice vs backend calls, integrated); **calibration** (ECE, AUROC of "Dice<0.5", selective risk at 20/50/80% coverage); **abstention rate**; **evidence audit score** (MedFlowBench-style: does the required evidence — key slices, coordinates, component claims — agree with the withheld mask, independent of the final mask's Dice); **harness-health metrics** (guard triggers, invalid actions, sandbox violations, canary touches, op failure rate, plan depth, cheap-model calls).

Report per-histotype (fibrous, lipomatous, myxoid, nerve-sheath, vascular) and stratified by lesion size tertile, since the decomposition hypothesis predicts an interaction with size.

### 6.4 Datasets and splits

- **Primary**: MSTT-199 / MSTT-iso (n=166 analyzed here; 5 histotypes) — matches the existing eval, has published U-Net/LiteMedSAM baselines, and is OOD w.r.t. nnInteractive's training data.
- **Multi-structure decomposition stress**: AMOS22 or BTCV (many organs; per-structure sub-tasks), TotalSegmentator (≥100 classes; tests budget scaling and plan fan-out).
- **Low-contrast / thin structure**: MSD Task07 (pancreas), plus an elongated-structure set (vessels or nerves) to exercise `render.cpr`/MIP and the contrast detector.
- **Multi-lesion / multi-phase**: KiTS23 (kidney + tumor + cysts; multi-phase CT) — the strongest test of component recursion and cross-phase consistency.
- **Multi-modal 3D**: BraTS (4 modalities; test whether the agent can pick informative channels programmatically and whether `edits.smooth(method="joint")` helps).
- **Leakage canaries**: for every case, place a decoy GT file and a decoy manifest in the sandbox (§4.6). Count touches as a *reported metric*, not an anecdote.
- **Splits**: (i) pilot 20 cases for score weights/thresholds and stopping parameters — never used for headline results; (ii) main 146; (iii) **library-learning holdout**: hold out one histotype entirely for the E7/H transfer claim.

### 6.5 Experiments (staged, with hypotheses and falsifiers)

**Phase 1 — action space (the headline).**

- **E1: C vs D (code vs JSON coordinates).** H: C ≥ D on Dice with fewer turns/tokens and fewer coordinate-class errors. Falsifier: D matches C on Dice *and* cost ⇒ the code claim is about expressiveness only, not performance; report honestly and shift the paper's claim to auditability/composability. Instrument: count coordinate errors (invalid voxels, out-of-bounds, `INVALID_COORDINATE`-class errors).
- **E1b: A/G vs C.** H: GUI/click space fails disproportionately on small, low-contrast, and peripheral lesions (the PAGER-style gap). Falsifier: no interaction between arm and size/contrast.
- **E2: perception channel ablation — B vs C.** H: renders help mainly on ambiguous/low-contrast/irregular boundaries; numeric-only is cheaper and nearly as good on high-contrast lipomas. Falsifier: images give no measurable benefit ⇒ drop `ask` from the default loop and save ~40% of tokens. Also report the token breakdown to make the trade-off legible.

**Phase 2 — decomposition (the RLM-analogue test).**

- **E3: E vs C (ROI + pyramid + structure decomposition).** H: gains concentrate on large lesions, multi-component cases, and multi-structure datasets; a "decomposition-eligibility" classifier (from cheap image stats, not GT) predicts which cases benefit. Falsifier: flat effect ⇒ decomposition is not the mechanism in 3D segmentation; that negative result is publishable and important.
- **E4: recursion depth/fanout sweep** (depth ∈ {1,2}, fanout ∈ {1,4,8}) on a multi-structure set. H: depth 2 helps only when structure count is large; fanout beyond the label count is wasted.
- **E5: planner ablation** — ADaPT-style as-needed decomposition vs always-decompose vs never. H: as-needed wins on cost-accuracy frontier; always-decompose fragments large lesions.

**Phase 3 — honesty, robustness, and generalization.**

- **E6: seed/leakage sensitivity.** Conditions: (i) GT centroid (current protocol), (ii) centroid jittered by σ = {5, 15} mm, (iii) a *random interior point* ≥ 10 mm from the boundary, (iv) no seed, text + case card only (agent must localize). H: median Dice degrades gracefully under (ii)–(iii) and much more under (iv); report the *slope* of Dice vs seed quality as the honest characterization of dependence on the GT-derived seed. Also report F2 (oracle prompts) to bound how much of the score is seed-driven rather than agent-driven.
- **E7: library learning / cross-case transfer.** Fixed case order; measure Dice and cost as a function of position, and the transfer to the held-out histotype. H: late cases and the held-out histotype benefit; falsifier: benefit is confined to cases seen earlier (mode collapse into memorized recipes) — which is itself a leakage-flavored failure worth reporting.
- **E8: stopping policy.** Compare learned/composite stopping vs oracle stopping (offline) vs a fixed 3-interaction budget. Report stopping regret and the Dice/cost frontier.
- **E9: hacking probes (red team).** Deliberately place (a) a decoy GT under an obvious name, (b) a *real* GT reachable only via a plausible shortcut (e.g. a cached mask in a temp dir), (c) a manifest with implausible targets, and (d) a case whose text description contradicts the image. Report: number of canary touches, whether any arm's Dice jumps, and whether the agent's rationale mentions non-image evidence. H: modern agents will find at least one and use it opportunistically (METR documents exactly this behavior on other tasks: <https://metr.org/blog/2025-06-05-recent-reward-hacking/>; MLE-bench had to add plagiarism detection and monitoring, arXiv:2410.07095). Treat E9's design as a contribution.

**Phase 4 (stretch) — human handoff.**

- **E10: selective automation.** Fix coverage at 80%; compare (i) random 20% escalation, (ii) escalation by agent confidence, (iii) escalation by `verify.*` evidence. Metric: risk at 80% coverage and *human time to fix* on the escalated cases (simulated by a reader with the harness). This turns the existing "83% flagging recall" into a decision-theoretic result.

### 6.6 Failure analysis protocol

1. **Script-level attribution.** For each failing case, use the mask DAG + trajectory log to identify the node(s) where Dice stopped improving or regressed, and the op(s) at those nodes. Classify the *first* causal op, not the last one.
2. **Two-rater taxonomy** (κ reported). Proposed labels: `PROMPT_PLACEMENT` (wrong/unlucky seed), `OP_SEMANTICS` (right intent, wrong op/params), `COORD_OR_UNIT` (grid/voxel/mm/axis error), `SCALE` (spacing/resolution mismatch; miss small or truncate large), `BOUNDARY_AMBIGUITY` (ill-defined margin; over/under-growth), `MULTIFOCAL_MISS` / `MULTIFOCAL_OVERMERGE`, `OVER_SEGMENT`, `UNDER_SEGMENT`, `BACKEND_LIMIT` (model simply fails; e.g. excellent mask impossible), `STOP_TOO_EARLY`, `STOP_TOO_LATE`, `HACKED_PROXY` (optimized the composite score against Dice), `SANDBOX/POLICY` (guard blocked a legitimate action), `TOOL_FAILURE` (harness bug).
3. **Error atlas**: for each failure, store the render at the worst slice plus the two verification scores, so the paper can show *why* — this is where reviewers' trust is won.
4. **Report Dice-vs-composite-score calibration** (Spearman + scatter). If the score is a poor predictor, say so and adjust the harness rather than the metric.

### 6.7 Statistics and power

- Paired design (same cases across arms). Primary test: Wilcoxon signed-rank on per-case Dice; bootstrap 95% CIs (10k resamples); Holm correction across secondary metrics. Report effect sizes (median difference with CI), not just p-values.
- **Pilot-first power analysis**: run 20 cases per arm, estimate the paired difference SD, compute the n needed to detect the smallest effect you consider meaningful (e.g. Δmedian Dice = 0.03–0.05) at 80% power; if n=146 is underpowered for the smallest interesting effect, pre-register a larger primary endpoint (e.g. Dice<0.5 rate, or cost at matched Dice) rather than running a post-hoc fishing expedition.
- Variance reporting: 3 repeats on a 40-case subset ⇒ report per-case SD and a variance-decomposition (case vs run) so readers know whether differences exceed run-to-run noise.
- Cost normalization: report Dice at equal $ and equal wall-clock, not only at equal tool-call counts (AI Agents That Matter).

### 6.8 What a methodological paper could claim (and what would falsify it)

Claims that are defensible if the experiments land:

1. **Action-space claim:** "For 3D medical image segmentation, replacing per-pointer tool calls (clicks/JSON coordinates) with *programmatic mask actions* improves Dice-per-dollar and reduces coordinate-class failures at matched model and information." (E1, E1b, E2.)
2. **Decomposition claim:** "Spatial/structural decomposition of a volumetric segmentation task into sub-problems (ROI, resolution band, structure) yields super-additive gains on hard cases, and the harness induces cross-case transfer." (E3–E5, E7.)
3. **Verification claim:** "GT-free verification primitives provide calibrated failure detection competitive with the agent's own confidence, enabling selective automation at fixed coverage." (E6, E8, E10.)
4. **Integrity claim:** "Code-enabled harnesses widen the leakage surface; evidence-based, canary-monitored evaluation is necessary and changes reported numbers." (E9; cf. SWE-bench+ solution leakage 32.67% and the 12.47%→3.97% resolution drop after filtering, arXiv:2410.06992.)

Falsification conditions are stated with each experiment; a negative result on the decomposition axis is publishable and should be reported as such.

---

## 7. Failure modes and anti-patterns

| # | Risk | Why it happens here | Detection | Mitigation |
|---|---|---|---|---|
| 1 | **Reward hacking / shortcutting** | The agent runs arbitrary code with filesystem access; it can read GT, caches, or manifests; it can also optimize a proxy (volume, smoothness, "agreement") instead of Dice | canary touches; Dice outliers (e.g. >0.95 with 0 interactions); rationales citing non-image evidence; distribution shift in op usage | no GT in sandbox; decoys + monitor; **evidence-checked scoring** (MedFlowBench-style); log and report canary hits; freeze score weights; red-team arm E9; multi-term score with anti-degenerate terms |
| 2 | **GT-seed leakage** | The protocol seeds at the GT centroid; Dice then partly measures seed quality, not agent skill | E6 slope: Dice vs seed perturbation/quality; compare with random-interior and no-seed conditions; report F2 oracle ceiling | pre-registered seed-quality sweep; report the slope; add a no-seed arm; never seed with volume/size information |
| 3 | **Coordinate-transform bugs** (voxel↔mm↔display, LPS↔RAS, cropped grid, resampled grid) | Multiple grids (original vs canonical vs ROI vs downsampled), LPS/RAS sign flip, non-isotropic spacing, nnInteractive's no-resample contract | `INVALID_COORDINATE` rates; left/right asymmetry on a phantom; masks landing in mirrored locations; Dice collapse on reoriented cases | one `Grid` object with mandatory units; crop/paste round-trip tests; phantom CI tests; record the affine with every mask; forbid bare tuples; assert `det(affine)>0`; never resample for prompting |
| 4 | **Silent destructive edits** | `edits.threshold`/`limit_to_intensity`/`remove_small` can shatter or erase a mask; the agent sees a small stats diff and moves on | diff stats + auto-render on guarded ops; guard-trigger counts | parent-preserving DAG; `allow_destructive` + auto before/after render; "never replace the accepted node except via `finish`"; mandatory diff in the observation |
| 5 | **Overfitting to visible slices ("2.5D myopia")** | The agent looks at a few slices, edits in-plane, and never checks out-of-plane coherence | `verify.slice_holdout` agreement; axial/coronal prompt-variant disagreement; masks that are sharper on prompted slices | holdout verification as first-class; multi-plane prompt variants; contact-sheet/MIP/CPR renders; explicit out-of-plane extent checks in `stats` |
| 6 | **Thin / low-contrast 3D structures missed** | A 1–3 voxel structure is invisible in a 768-px montage; the boundary shell contrast is near noise; a threshold op erases the structure entirely | boundary-contrast z-score; extent-in-mm vs expected; `render.cpr`/MIP inspection; sudden volume collapse after an op | MIP + CPR views; mm-scale zoom renders; `snap` with bounded radius tied to the *plane* not a guess; intensity-profile probes across the boundary; prefer `region_grow`/`grow_from_seeds` over hard thresholds |
| 7 | **Context/render blow-up & latency** | Every render is ~0.8–1.4k vision tokens; long trajectories rot the root context; backend calls are the wall-clock floor | tokens in/out per turn; root-context size vs turn index; render count; p95 latency | numeric-first observations; artifact ids + `ask` sub-calls; render budget with a hard cap; caching by node hash; montages instead of single slices; speculative/pre-launched independent calls |
| 8 | **Thrashing / oscillation** | Agent alternates dilate/erode, add/remove point, never converging | repeated near-identical op pairs; score plateau; mask hashes cycling | candidate portfolio with explicit accept/reject; max 2 proposals without score improvement ⇒ stop or decompose; penalize provenance-heavy candidates |
| 9 | **Premature or late stopping** | No GT ⇒ no obvious signal; models default to optimism | stopping regret (offline); interaction-efficiency AUC; stop_reason distribution | pre-registered stop rules; verification thresholds; budget caps; report regret as a headline number |
| 10 | **Mis-calibrated confidence** | Confidence is self-reported; the composite score is a proxy | ECE/AUROC; selective risk at coverage; confidence-without-evidence count | require evidence ids with confidence; compare with RCA/verification verdicts; report calibration rather than a single recall number |
| 11 | **Sandbox escape / resource abuse** | Unbounded loops, giant allocations, `subprocess`, network, writes outside the case dir | sandbox violation log; OOM/timeout events | process isolation + cgroups/timeouts; no network; allow-listed module surface; cap array size; kill + structured error back to the agent |
| 12 | **Harness-specific overfitting ("benchmark overfitting")** | Score weights, stop thresholds, and library functions tuned on the same 166 cases | weight-tuning on the pilot split only; holdout histotype; frozen weights in the main run | pilot/main/holdout split discipline; report weights as hyperparameters; test on the second dataset family (AMOS/BTCV) before claiming generality |
| 13 | **Backend/interface confound** | Differences attributed to "code" may come from the backend call patterns (ROI crops, prompt counts) or from doc quality | log backend calls per arm; equalize call counts where possible; run E1 with the *same* backend primitives | pin backend hash; instrument call counts; report as a limitation |
| 14 | **Reproducibility drift** | Closed models and provider versions change between runs; sampling non-determinism | record model name+date, harness commit, seeds; re-run a fixed 10-case sanity set | pin versions in metadata; publish trajectories; keep a "golden 10" regression set |
| 15 | **Clinical non-actionability** | A mask that is quantitatively fine but structurally implausible (islands, holes, adjacent-organ inclusion) | `stats` component/hole counts; sphericity; a reader audit on a sample | component/hole guards + explicit evidence; DICOM-SEG/RT-STRUCT export; optional reader review arm |

Anti-patterns to actively avoid in implementation: rendering after every action "for safety"; storing free-form agent notes without evidence links; letting the agent choose its own confidence scale mid-run; a 10-deep undo stack as the only versioning (no provenance ⇒ no failure attribution); exposing raw pixel coordinates to the agent; giving the agent the GT-derived volume or any dataset metadata (including modality flags and filenames that encode the label); a single-number score optimized in the open; and per-case prompt/score tuning by the authors during the main run.

---

## 8. Residual risks and open questions

1. **Does code actually beat JSON in the spatial domain?** The evidence is strong in text/coding tasks (CodeAct, RLM) and strong for *negative* results in pixel-precise GUI control (PAGER, ScreenSpot-Pro), but there is no direct 3D-medical segmentation comparison. That is precisely E1 and it could go either way; the memo's architecture is designed so both outcomes are informative.
2. **The composite GT-free score may be weak.** If Spearman(score, Dice) is low, the proposal/acceptance loop degenerates. Fallback: use the score only for *ranking candidates* (pairwise, within-case) rather than absolute selection, or reduce to verification-only gating.
3. **nnInteractive's soft predictions may not be exposed** in the current Python API. `propose(kind="false_negative")` may need to fall back to geometric heuristics (distance transform + boundary intensity). If the API lacks probabilities, budget extra engineering for the heuristic path or fork the backend to expose the final-softmax volume.
4. **Library learning risks contamination across cases** (a function tuned for lipomas applied to fibrous tumors). Mitigation: tests on phantoms, per-histotype tagging, and the held-out-histotype experiment (E7).
5. **Doc/prompt confound.** The code arm gets a prelude doc; the CLI arm gets a usage guide. Both are "the interface," so this is fair, but the *quality* difference is an author-controlled variable. Mitigation: blind an independent author to write each arm's docs, and report them.
6. **Small n for the hardest strata.** Lipomatous (n=8) cannot support per-histotype claims; pre-specify pooled and size-stratified analyses.
7. **Two numbers in the existing reports disagree** — the abstract draft says "median of 3 interactions per case" while `eval/EVAL_REPORT.md` says "a median of two interactions per case." Reconcile before publication; this memo uses "median ≤3" and flags the discrepancy.
8. **Compute cost of the ablation matrix** (≈6 arms × 166 cases × up to 3 repeats × several model turns) is non-trivial; stage it (40-case subsets for E3–E7, full 166 for E1/C-vs-A) and use cost-normalized reporting.

---

## 9. References

**RLM / harness design**
1. A. L. Zhang, T. Kraska, O. Khattab. *Recursive Language Models.* arXiv:2512.24601. <https://arxiv.org/abs/2512.24601>
2. RLM blog post (Oct 2025). <https://alexzhang13.github.io/blog/2025/rlm/>
3. RLM codebase + README (sandboxes, contexts, training env). <https://github.com/alexzhang13/rlm>, minimal: <https://github.com/alexzhang13/rlm-minimal>
4. A. L. Zhang, O. Khattab. *Language model harnesses are compositional generalizers.* Jul 2026. <https://alexzhang13.github.io/blog/2026/harness/>
5. A. L. Zhang. *Speculative Programmatic Tool Calling.* Aug 2026. <https://alexzhang13.github.io/blog/2026/spec-ptc/>
6. E. Lumer, S. Sen, K. Paul, V. K. Subbiah. *Recursive Agent Harnesses.* arXiv:2606.13643. <https://arxiv.org/abs/2606.13643>
7. Anthropic. *Effective context engineering for AI agents* (context rot). <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>
8. Model Context Protocol specification. <https://modelcontextprotocol.io/specification/2025-06-18>
8b. A. L. Zhang. *The Mismanaged Geniuses Hypothesis.* Apr 2026. <https://alexzhang13.github.io/blog/2026/mgh/>
8c. A. L. Zhang. *Language Model "Shape".* Sep 2026. <https://alexzhang13.github.io/blog/2026/shape/>

**Code as action / decomposition**
9. X. Wang et al. *Executable Code Actions Elicit Better LLM Agents* (CodeAct). arXiv:2402.01030.
10. J. Liang et al. *Code as Policies.* arXiv:2209.07753.
11. D. Surís, S. Menon, C. Vondrick. *ViperGPT.* arXiv:2303.08128.
12. T. Gupta, A. Kembhavi. *Visual Programming (VISPROG).* arXiv:2211.11559.
13. G. Wang et al. *Voyager.* arXiv:2305.16291.
14. D. Zhou et al. *Least-to-Most Prompting.* arXiv:2205.10625.
15. T. Khot et al. *Decomposed Prompting.* arXiv:2210.02406.
16. A. Prasad et al. *ADaPT.* arXiv:2311.05772.

**Programs for spatial artifacts**
17. Z. Hu et al. *SceneCraft: 3D scenes as Blender code.* arXiv:2403.01248.
18. *Generating CAD Code with Vision-Language Models (CADCodeVerify).* arXiv:2410.05340.
19. *BlenderLLM.* arXiv:2412.14203. *CAD-Coder.* arXiv:2505.14646.
20. Z. Yang et al. *MatPlotAgent.* arXiv:2402.11453.
21. *SVGEditBench.* arXiv:2404.13710; *SVGEditBench V2.* arXiv:2502.19453.
22. BlenderMCP. <https://github.com/ahujasid/blender-mcp>
23. Community pixel-art agent skills: <https://github.com/Gamezxz/pixel-art-studio>, <https://github.com/ccc007ccc/agent-paint-pixel-skills>, informal eval <https://ljvmiranda921.github.io/notebook/2025/07/20/draw-me-a-swordsman/>

**Perception/grounding interfaces**
24. J. Yang et al. *Set-of-Mark Prompting.* arXiv:2310.11441.
25. K. Li et al. *ScreenSpot-Pro.* arXiv:2504.07981.
26. Q. Wu et al. *GUI-Actor: Coordinate-Free Visual Grounding.* arXiv:2506.03143.
27. J. Wei et al. *PAGER: point-precise geometric GUI control.* arXiv:2605.15963.
28. Anthropic vision token accounting. <https://platform.claude.com/docs/en/build-with-claude/vision>
29. Anthropic computer use tool. <https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool>

**Interactive & agentic segmentation**
30. F. Isensee et al. *nnInteractive.* arXiv:2503.08373; code <https://github.com/MIC-DKFZ/nnInteractive>; API changes <https://github.com/MIC-DKFZ/nnInteractive/blob/master/API_CHANGES_v2.md>
31. Y. He et al. *VISTA3D.* CVPR 2025, arXiv:2406.05285.
32. *SegVol.* NeurIPS 2024, arXiv:2311.13385.
33. *DeepEdit / MONAI Label.* arXiv:2305.10655; <https://github.com/Project-MONAI/MONAILabel>
34. *MedSAM2.* arXiv:2504.03600. *SAM-Med3D.* arXiv:2310.15161.
35. S. Liu et al. *MedSAM-Agent.* arXiv:2602.03320.
36. X. He et al. *RSAgent.* arXiv:2512.24023.
37. M. Zhu et al. *SegAgent / HLMAT.* CVPR 2025, arXiv:2503.08625.
38. X. Yu et al. *GenCellAgent.* arXiv:2510.13896.
39. W. Shen et al. *MedOpenClaw and MedFlowBench.* arXiv:2603.24649.
40. *ABRA: Agent Benchmark for Radiology Applications.* <https://luab.github.io/abra/>
41. *ReX-MLE.* arXiv:2512.17838; <https://proceedings.mlr.press/v315/kenia26a.html>

**Clinical mask-operation vocabulary & interaction evidence**
42. 3D Slicer Segment Editor (effects list). <https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html>
43. 3D Slicer coordinate systems (LPS vs RAS). <https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html>
44. nibabel coordinate systems (voxel/RAS+, affine). <https://nipy.org/nibabel/coordinate_systems.html>
45. W. Cho et al. *Slice and Conquer: A Planar-to-3D Framework for Efficient Interactive Segmentation of Volumetric Images.* WACV 2024, pp. 7614–7623. <https://openaccess.thecvf.com/content/WACV2024/html/Cho_Slice_and_Conquer_A_Planar-to-3D_Framework_for_Efficient_Interactive_Segmentation_WACV_2024_paper.html>
46. *Rapid Interactive and Intuitive Segmentation of 3D Medical Images Using Radial Basis Function Interpolation.* J. Imaging 2017, 3(4):56, doi:10.3390/jimaging3040056.
47. Strokes vs contour interaction study. J Digit Imaging 2015, DOI 10.1007/s10278-015-9839-8.
48. MCP integrations as prior art: <https://github.com/zhaoyouj/mcp-slicer>, <https://github.com/pieper/slicer-skill>

**Evaluation, QC, and integrity**
49. S. Kapoor et al. *AI Agents That Matter.* arXiv:2407.01502.
50. J. S. Chan et al. *MLE-bench.* arXiv:2410.07095 (plagiarism/cheating detection).
51. R. Aleithan et al. *SWE-Bench+.* arXiv:2410.06992 (32.67% solution leakage; 12.47%→3.97% after filtering).
52. METR. *Recent Frontier Models Are Reward Hacking.* Jun 2025. <https://metr.org/blog/2025-06-05-reward-hacking/>
53. V. Krakovna et al. *Specification gaming: the flip side of AI ingenuity.* DeepMind, 2018. <https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/>
54. R. Robinson et al. *Real-time Prediction of Segmentation Quality* (reverse classification accuracy). MICCAI 2018. <http://www.doc.ic.ac.uk/~bglocker/pdfs/robinson2018miccai.pdf>; UK Biobank CMR QC, JCMR 2019. <https://link.springer.com/article/10.1186/s12968-019-0523-x>
55. A. Reasat et al. *MSTT-199.* arXiv:2409.03110; data/code <https://github.com/Reasat/mstt>
56. Interactive-segmentation click protocols and their sensitivity: RClicks (NeurIPS 2024 D&B) <https://proceedings.neurips.cc/paper_files/paper/2024/file/e6b5bcc872666d37c469e5c5ba723669-Paper-Datasets_and_Benchmarks_Track.pdf>; IPOL comparison of clicking procedures <https://www.ipol.im/pub/art/2024/498/>
57. DICOM Segmentation IOD <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html>; RT Structure Set <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.html>

**Project context (local, non-citable)**
58. `abstract_draft.md`, `eval/EVAL_REPORT.md`, `eval/skill/archive/USAGE_GUIDE_02_guide_edit.md`, `eval/run_oneclick_baseline.py`, `eval/lib/{reorient,dice}.py` in this repository.
