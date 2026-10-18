# Potential Plan 1: A Code-First Harness for Agentic 3D Segmentation

I think your central intuition is right, with one important refinement:

> The best interface is probably not unrestricted Python that manipulates voxel arrays directly. It is a **sandboxed Python environment with a typed volumetric-segmentation prelude**.

That gives coding agents the medium they are strongest in—program composition, loops, conditionals, inspection, debugging, and reusable functions—without inviting silent coordinate errors, ground-truth leakage, arbitrary filesystem access, or enormous NumPy operations.

The DeepSeek memo strongly supports this direction and is saved at:

`research/HARNESS_DESIGN_RESEARCH.md`

## 1. The RLM analogy maps unusually well to volumetric segmentation

The key idea from Alex Zhang's [Recursive Language Models](https://arxiv.org/abs/2512.24601) and his later [harness essay](https://alexzhang13.github.io/blog/2026/harness/) is not recursion by itself. It is:

1. Put the large object in an environment as a variable.
2. Let the model inspect and transform it programmatically.
3. Keep task-specific bulk data out of the root context.
4. Decompose an OOD global task into locally familiar subproblems.
5. Make each individual model call locally in-distribution even if the overall task is OOD.

For segmentation:

| RLM concept | Segmentation equivalent |
|---|---|
| Long context as a Python variable | Image volume and masks as persistent handles |
| Peek/search/split context | Crop ROIs, summarize components, inspect selected slices |
| Recursive call on a chunk | Child task on a subvolume, structure, component, or boundary band |
| Keep intermediate results in variables | Keep masks, statistics, renders, and plans in an external store |
| Small root context | Compact case card plus numeric observations |
| Programmatic submodel calls | Ask a vision model a focused question about one cached render |

The natural recursive units are not arbitrary slices. They are:

- a suspicious subvolume;
- one lesion or connected component;
- one anatomical structure in a multilabel task;
- a coarse-resolution segmentation followed by a fine boundary-band task;
- an uncertainty region where competing masks disagree.

That is the most compelling connection to the RLM/OOD argument: a 512×512×300 tumor volume may be unfamiliar globally, but “decide whether this candidate component belongs to the target,” “repair the posterior boundary in this 60 mm crop,” and “remove disconnected vessel-like voxels” are small, repeated, structurally similar tasks.

## 2. Recommended shape: typed gate, code interior

DeepSeek summarized the architecture well:

> **Code is the work surface; typed function calls are the gate; screenshots are observations rather than the artifact of record.**

I would build six layers.

### A. Case and state store

The agent receives a compact case card:

```json
{
  "case_id": "MSTT_F17",
  "modality": "T2 MRI",
  "target": "soft-tissue tumor",
  "grid_id": "grid_native",
  "shape": [384, 384, 112],
  "spacing_mm": [0.78, 0.78, 3.0],
  "orientation": "RAS+",
  "seed": {"mm": [42.1, -18.4, 71.0]},
  "budgets": {
    "model_turns": 20,
    "backend_calls": 12,
    "renders": 6
  }
}
```

The image and masks remain in the environment as handles. They are never serialized into the language context.

### B. Sandboxed Python REPL

The agent writes short programs against an allowlisted package. It cannot use:

- `open`
- `os`
- `subprocess`
- arbitrary imports
- network access
- unrestricted filesystem paths

The REPL can access only the current image, masks, approved tools, and scratch space.

### C. Typed segmentation prelude

The API should express operations at the semantic level coding agents understand well.

```python
m1 = seg.point(
    mask=None,
    point=seed,
    label="positive",
    backend="nninteractive",
    coordinate_space="mm",
)

s = stats(m1)
candidate_points = propose(m1, region="likely_false_negative", k=5)

m2 = seg.point(
    mask=m1,
    point=candidate_points[0],
    label="positive",
    backend="nninteractive",
)

m3 = edits.keep_component(m2, containing=seed)
m4 = edits.close(m3, radius_mm=2.0)
```

The runtime resolves coordinates and executes operations. The agent should generally choose candidate points or regions produced by the harness rather than hallucinating raw coordinates.

### D. Immutable mask DAG

Every operation creates a new mask version:

```text
m0
└── m1: nnInteractive positive seed
    ├── m2: positive point at candidate_3
    │   └── m4: close(radius_mm=2)
    └── m3: tight-box restart
```

Each node stores:

- parent mask hash;
- operation and arguments;
- code that produced it;
- grid and affine;
- execution time;
- voxel and physical-volume differences;
- render IDs;
- tool/backend version;
- model turn that requested it.

No in-place mutation. The agent can compare, branch, roll back, and commit:

```python
compare(m2, m4)
checkout(m2)
finish(m4, confidence="medium", evidence=[...])
```

This is dramatically better than a ten-step undo stack because it supports failure analysis, replay, oracle stopping analysis, and counterfactual comparison.

### E. Numeric-first feedback and rendering service

The default observation should not be another screenshot. Return structured diagnostics:

```python
stats(m4)
```

```json
{
  "volume_ml": 41.3,
  "components": 2,
  "component_volumes_ml": [40.9, 0.4],
  "bbox_mm": [57, 81, 66],
  "centroid_mm": [42.0, -18.1, 70.6],
  "slices_with_mask": {
    "axial": [36, 66],
    "coronal": [121, 193],
    "sagittal": [142, 211]
  },
  "inside_intensity": {"p05": 83, "p50": 142, "p95": 211},
  "outside_shell_intensity": {"p05": 48, "p50": 92, "p95": 167},
  "change_from_parent": {
    "added_ml": 2.1,
    "removed_ml": 0.3
  }
}
```

Only render when visual inspection is likely to change the next action:

```python
r = render(
    m4,
    view="orthogonal",
    center="largest_change",
    overlay="outline",
    include=["parent_diff", "prompts"],
)
```

The agent receives an artifact ID plus render metadata. A focused vision subcall can inspect the render without polluting the root context:

```python
ask(
    r,
    "Does the posterior boundary follow the visible tumor margin? "
    "Answer complete, undersegmented, oversegmented, or uncertain."
)
```

This is the direct RLM analogue: the root model invokes a model over a bounded visual “chunk,” then receives a compact answer.

### F. External controller

A controller outside generated code enforces:

- budgets;
- valid operations;
- time and memory limits;
- destructive-edit guards;
- final-mask schema;
- stopping rules;
- evidence requirements;
- ground-truth isolation.

The model cannot override these controls.

## 3. “Painting” should be programmatic geometry, not voxel emission

You do not want the agent to generate a list of painted voxels. That recreates coordinate prediction at an even worse scale.

The most useful primitives are high-level, parametric edits familiar from 3D Slicer and ITK-SNAP:

### Learned proposal operations

```python
seg.point(...)
seg.box(...)
seg.scribble(...)
seg.lasso(...)
seg.text(...)
seg.restart(...)
```

### Mask algebra

```python
union(a, b)
intersect(a, b)
subtract(a, b)
xor(a, b)
```

### Component operations

```python
keep_component(mask, containing=seed)
remove_small(mask, min_volume_ml=0.5)
split_components(mask)
component_nearest(mask, landmark)
```

### Physical morphology

```python
dilate(mask, radius_mm=2)
erode(mask, radius_mm=1)
open(mask, radius_mm=1.5)
close(mask, radius_mm=2)
fill_holes(mask, max_volume_ml=1)
smooth(mask, sigma_mm=0.8)
```

All radii should be in millimetres, not voxels.

### Intensity-aware operations

```python
region_grow(seed, tolerance_sigma=1.5)
restrict_intensity(mask, percentiles=(5, 98))
threshold_within(mask_or_roi, lower, upper)
boundary_snap(mask, method="gradient")
```

### Volumetric painting primitives

```python
paint_sphere(center_mm, radius_mm)
paint_tube(path_mm, radius_mm)
paint_slab(axis, range_mm)
paint_between_contours(contours)
extrude_lasso(contour, distance_mm)
fill_between_slices(mask, axis="axial")
erase_bbox(lo_mm, hi_mm)
erase_component(component_id)
```

These let an agent “paint” in 3D through code while relying on the runtime for precise rasterization.

A particularly promising primitive is a **tube or swept brush along a spline**. Coding agents are likely much better at identifying a centreline and radius profile than manually placing hundreds of 3D brush strokes.

## 4. Coordinates must be treated as a type system

This may be the most important engineering requirement.

You have at least three coordinate systems:

- native voxel indices;
- physical patient coordinates in millimetres;
- displayed pixels in axial/coronal/sagittal renders.

Every function must require an explicit coordinate space:

```python
seg.point(point=(120, 140, 75), in_="voxel")
seg.point(point=(42.1, -18.4, 71.0), in_="ras_mm")
```

Bare triples should be rejected.

Every crop creates a child grid carrying explicit transformations back to the parent. Every mask is associated with one grid ID. Boolean operations between different grids should fail unless an explicit resampling or projection operation is requested.

This matters especially because:

- DICOM typically uses LPS;
- NIfTI tooling often uses RAS+;
- nnInteractive uses the original image grid and does not silently resample;
- cropped and isotropically resampled views introduce additional transforms.

A left/right flip or crop-offset bug can look like agent failure when it is actually harness failure.

## 5. Proposed segmentation loop

I would make the harness encourage this fixed sequence:

### 1. Localize

- Read the case card.
- Inspect a low-resolution volume summary or montage.
- Identify a target ROI.
- Decide whether the supplied seed is credible.

### 2. Propose

- Call nnInteractive with a point, box, scribble, or lasso.
- Optionally generate several prompt variants concurrently.
- Store all resulting candidates.

### 3. Characterize

- Measure components, volume, extent, intensity contrast, and boundary properties.
- Identify suspicious regions.
- Compare prompt variants.

### 4. Decompose if necessary

- Crop the lesion ROI.
- Split components.
- Create a child task for one uncertain boundary or satellite lesion.
- Use a resolution pyramid: coarse extent first, boundary shell second.

### 5. Repair

- Add a positive/negative prompt.
- Apply a mask operation.
- Create a new candidate, never overwrite the accepted candidate.

### 6. Verify

Use multiple independent checks:

- prompt-variant agreement;
- withheld-slice consistency;
- component and topology checks;
- boundary contrast;
- candidate comparison;
- focused visual review of high-change slices.

### 7. Commit, abstain, or escalate

The agent must finish with:

- selected mask version;
- confidence;
- evidence IDs;
- reason for stopping;
- unresolved ambiguity;
- whether human review is requested.

## 6. Why the hybrid should beat pure GUI or pure JSON

| Action space | Advantages | Main failure |
|---|---|---|
| Screenshot + click | Simple, human-like | Current VLMs are bad at precise coordinates and viewer-state management |
| JSON function calls | Typed and safe | One call per edit; poor composition; agent still guesses coordinates |
| Code over volumes | Composable, testable, reusable, low round-trip cost | Much larger leakage and safety surface |
| **Typed gate + code interior** | Composition plus safety and auditability | More engineering work |

The existing prototype already demonstrates the GUI problem: it introduced a `--snap` heuristic because image-derived coordinates could be off by 5–20 pixels. That is evidence that the action space is forcing the agent to solve the wrong problem.

The agent should decide:

> “Select the largest likely false-negative component and add a positive prompt near its medial axis.”

It should not decide:

> “Click pixel 433, 281 in this resized screenshot.”

The harness can calculate the point exactly.

## 7. The strongest methodological experiment

The paper should hold the model, data, seed, backend, budgets, and information fixed while varying only the harness/action space.

### Core arms

| Arm | Interface |
|---|---|
| A | Existing screenshot/CLI interface |
| B | JSON calls with point/box/scribble coordinates |
| C | Code/DSL, numeric feedback only |
| D | Code/DSL plus selective renders |
| E | D plus explicit ROI/resolution/component decomposition |
| F | One-click nnInteractive baseline |
| G | Oracle-prompt ceiling, reported separately |

The key comparisons are:

1. **Code versus JSON**: does composition improve Dice, cost, or coordinate-error rate?
2. **Visual feedback versus numeric-only**: when do images add value?
3. **Decomposition versus flat code loop**: does RLM-style structure actually improve OOD performance?
4. **Agent improvement over initial proposal**: how much does the agent add beyond the seed and nnInteractive?

Metrics should include:

- Dice, NSD, and HD95;
- improvement over the initial mask;
- Dice after each backend interaction;
- area under the Dice-versus-interaction curve;
- wall time, model turns, render count, tokens, and cost;
- invalid-coordinate and transform errors;
- stopping regret: best intermediate Dice minus final Dice;
- failure-detection AUROC and calibration;
- sandbox or leakage violations.

## 8. The seed issue must become an experiment, not just a limitation

Your current GT-centroid seed is a partial oracle. That is acceptable as one condition, but it cannot be the only condition if the paper claims autonomous or OOD segmentation.

Use at least:

1. Ground-truth centroid
2. Centroid jittered by 5 mm
3. Centroid jittered by 15 mm
4. Random interior point
5. Automatically localized point
6. No seed: text and image only

Report the slope of performance degradation as localization quality worsens. This tells readers whether the agent is segmenting or merely cleaning up an unusually good initialization.

It also directly tests the harness hypothesis: code/decomposition should matter more when the initial prompt is imperfect.

## 9. Your most defensible novelty claim

The broader claim “first 3D segmentation agent” is now probably too broad. The literature includes LORE, MedVol-R1, MedSegAgent, VILA-M3, and other systems that use true 3D backends or learn 3D policies.

A narrower and stronger claim would be:

> A code-first, training-free harness for true volumetric segmentation in which a general coding agent decomposes the task spatially, operates on persistent versioned masks through programmatic learned and classical tools, and verifies its work using low-latency numeric and visual feedback.

The methodological claims would then be:

1. **Programmatic mask actions outperform coordinate-centric interaction** at matched agent, backend, and information.
2. **Spatial decomposition improves OOD 3D segmentation**, particularly for large, multifocal, or structurally complex targets.
3. **External mask state and compact observations reduce context and latency** without sacrificing quality.
4. **Evidence-backed verification improves failure detection and stopping.**

That is a clearer contribution than simply integrating Claude with nnInteractive.

## 10. My recommended MVP

I would not start by building the full recursive multi-agent system.

Start with:

1. A `Grid` class with explicit voxel/mm/display transforms and extensive phantom tests
2. Immutable mask versions and replay
3. A sandboxed Python prelude with approximately 20 operations
4. nnInteractive point, box, scribble, and lasso adapters
5. `stats`, `compare`, `propose`, and orthogonal/difference rendering
6. A destructive-operation guard
7. Structured `finish(mask, confidence, evidence)`
8. A runner that reproduces the current MSTT experiment
9. Four arms:
   - existing CLI;
   - JSON tools;
   - code with numeric feedback;
   - code with numeric plus visual feedback

Only after showing that the action space matters would I add:

- ROI child tasks;
- focused vision subcalls;
- reusable function libraries;
- prompt-variant verification;
- multi-structure fan-out;
- online library learning.

That staged approach gives you a clean paper even if the recursive/decomposition layer does not help. The code-action result can stand alone; the decomposition result becomes the second contribution rather than a prerequisite.

---

## 11. Additional implementation principles

The following principles are worth fixing before implementation so that the harness remains both scientifically interpretable and safe.

### 11.1 Separate the action language from the implementation language

Python should be the agent-facing composition language, but operations should compile into a constrained intermediate representation before execution. For example:

```python
m2 = edits.close(m1, radius_mm=2.0)
```

could compile into:

```json
{
  "op": "morphology.close",
  "inputs": ["mask:m1"],
  "parameters": {"radius_mm": 2.0},
  "grid": "grid_native",
  "destructive": false
}
```

This provides Python's compositional advantages while retaining validation, deterministic replay, and language-independent logs.

### 11.2 Treat the harness as the experimental intervention

The model and segmentation backend should be held fixed wherever possible. Otherwise, any apparent improvement could be attributed to a stronger VLM, a different checkpoint, more prompts, or additional information. The central scientific question is whether the harness changes performance.

Every trajectory should therefore record:

- agent model and version;
- exact system and task prompts;
- harness commit;
- backend checkpoint hash;
- input image hash;
- seed and seed-generation method;
- all code and compiled operations;
- all rendered observations;
- timing, token, and cost information;
- every candidate mask hash;
- final and best-intermediate metrics computed after the run.

### 11.3 Preserve a human-review path

A code-first agent interface does not eliminate the value of a conventional viewer. The viewer should become another client of the same mask store. Human edits should create versioned nodes with the same provenance as agent edits. This supports:

- human-versus-agent action comparisons;
- selective escalation;
- measurement of correction time;
- replayable adjudication;
- eventual clinical workflow studies.

### 11.4 Build the analysis and replay system before the main experiment

The mask DAG and trajectory replay tools should exist before collecting the primary data. They should calculate:

- Dice, NSD, and HD95 for every candidate mask;
- the best candidate reached during a trajectory;
- stopping regret;
- improvement or regression caused by each operation;
- backend calls and render usage;
- invalid or guarded actions;
- failure type at the first causal error;
- whether a later rollback recovered from that error.

Without this system, the project would report only final masks and lose the strongest evidence about *why* the harness works.

### 11.5 Pre-register falsification conditions

Examples:

- If JSON function calls match code on quality and cost, the code-first performance claim is unsupported; the remaining contribution is auditability and composition.
- If selective rendering does not improve low-contrast or irregular cases, numeric feedback should remain the default and rendering should be minimized.
- If decomposition does not improve complex cases, the RLM analogy is conceptually interesting but not empirically supported in this domain.
- If the GT-free verification score does not correlate with Dice, it should not drive acceptance or stopping.
- If seed perturbation causes catastrophic degradation across all interfaces, localization—not segmentation—is the dominant unsolved problem.

Stating these conditions in advance will substantially strengthen the methodological credibility of the work.

## 12. Selected references

1. Zhang AL, Kraska T, Khattab O. [Recursive Language Models](https://arxiv.org/abs/2512.24601).
2. Zhang AL, Khattab O. [Language model harnesses are compositional generalizers](https://alexzhang13.github.io/blog/2026/harness/).
3. Wang X et al. [Executable Code Actions Elicit Better LLM Agents (CodeAct)](https://arxiv.org/abs/2402.01030).
4. Liang J et al. [Code as Policies](https://arxiv.org/abs/2209.07753).
5. Surís D, Menon S, Vondrick C. [ViperGPT](https://arxiv.org/abs/2303.08128).
6. Gupta T, Kembhavi A. [VISPROG](https://arxiv.org/abs/2211.11559).
7. Isensee F et al. [nnInteractive](https://arxiv.org/abs/2503.08373); [code](https://github.com/MIC-DKFZ/nnInteractive).
8. Liu S et al. [MedSAM-Agent](https://arxiv.org/abs/2602.03320).
9. Zhu M et al. [SegAgent](https://arxiv.org/abs/2503.08625).
10. Shen W et al. [MedOpenClaw and MedFlowBench](https://arxiv.org/abs/2603.24649).
11. Kapoor S et al. [AI Agents That Matter](https://arxiv.org/abs/2407.01502).
12. 3D Slicer. [Segment Editor effects](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html).
13. 3D Slicer. [Coordinate systems](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html).
14. NiBabel. [Coordinate systems and affines](https://nipy.org/nibabel/coordinate_systems.html).
15. Cho W et al. [Slice and Conquer: A Planar-to-3D Framework for Efficient Interactive Segmentation](https://openaccess.thecvf.com/content/WACV2024/html/Cho_Slice_and_Conquer_A_Planar-to-3D_Framework_for_Efficient_Interactive_Segmentation_WACV_2024_paper.html).

For the fuller research synthesis, citations, proposed API, detailed experimental matrix, failure taxonomy, and safety plan, see `research/HARNESS_DESIGN_RESEARCH.md`.

---

# Extension: Semantic Intent to Executable Coordinates

This is an active research problem, and our experience closely matches what has been observed in GUI agents and visual-grounding systems:

> A VLM can often identify the intended semantic region while still being unreliable at serializing that intention into exact numeric coordinates.

That distinction—**semantic localization versus coordinate emission**—is important. The latter should probably not be the LLM's responsibility.

## 13. Relevant research directions

### 13.1 Coordinate-free visual grounding

[GUI-Actor](https://arxiv.org/abs/2506.03143) explicitly argues that formulating grounding as text-based coordinate generation creates several problems:

- weak spatial–semantic alignment;
- ambiguous supervision when many points inside a region would be valid;
- mismatch between dense pixel coordinates and coarse visual patch features;
- poor transfer across display resolutions and layouts.

Instead of asking the model to emit `(x, y)`, GUI-Actor adds an action head that attends to relevant image patches and proposes one or more **action regions**. A verifier then selects among the candidate regions.

This maps almost exactly onto our problem:

```text
LLM intent: "place a positive point in the middle of the IPFP"
                         ↓
Wrong abstraction: model emits (437, 282)
                         ↓
Better abstraction: grounding module identifies an IPFP region
                         ↓
Runtime computes a valid maximally interior point
```

The LLM chooses the semantic target and operation. A spatial grounding component converts that into executable geometry.

### 13.2 Set-of-Mark prompting

[Set-of-Mark prompting](https://arxiv.org/abs/2310.11441) is probably the easiest immediate intervention.

The system:

1. Generates candidate regions using SAM, SEEM, MaskDINO, superpixels, or another proposal method.
2. Places visible identifiers over those regions: `1`, `2`, `3`, etc.
3. Asks the VLM to select an identifier rather than generate coordinates.
4. Maps the selected identifier back to the exact candidate region.

For our application, this could operate at several levels:

- candidate anatomical regions;
- candidate connected components;
- candidate point locations;
- candidate boundary-error regions;
- candidate slices or subvolumes.

Instead of asking:

> Give the pixel coordinates corresponding to the middle of the IPFP.

show a render with nine candidate points and ask:

> Which marker lies centrally within the infrapatellar fat pad?

The answer becomes `candidate_id=4`, and the harness already knows candidate 4 corresponds to native voxel `(131, 187, 42)` or RAS coordinate `(24.1, -17.6, 61.4)`.

This changes a fragile regression problem into a much easier discrete-selection problem.

### 13.3 Models trained explicitly to point

[Molmo/PixMo](https://arxiv.org/abs/2409.17146) demonstrates that pointing is a distinct capability that benefits from dedicated training data. PixMo-Points was collected specifically to connect natural-language requests to image points.

That is instructive for two reasons:

1. General visual understanding does not automatically imply reliable pointing.
2. If we eventually want a learned medical grounding module, we may need medical pointing trajectories rather than relying on a general VLM.

We could potentially collect these trajectories cheaply from existing masks. For any structure mask, derive multiple valid point targets:

- maximal-inscribed point;
- centroid projected into the mask;
- superior/inferior/anterior/posterior portions;
- boundary-error regions;
- points near or far from named anatomical landmarks.

The model can then learn concepts like “central,” “anterior margin,” and “inferolateral component” in medical coordinates.

### 13.4 Medical image grounding

There is now a meaningful line of work on mapping medical language to boxes and masks:

- [VividMed](https://aclanthology.org/2025.naacl-long.89/) supports medical phrase grounding into masks and bounding boxes, including 2D and 3D data.
- [M3D](https://arxiv.org/abs/2404.00578) includes 3D positioning and referring-expression segmentation tasks.
- [MedGround-R1](https://github.com/bio-mlhui/MedGround-R1) learns medical image grounding with spatial and semantic rewards and a “Chain-of-Box” representation.
- Grounding-DINO→SAM and related general-domain systems translate language into boxes and then masks.
- Models such as SAT, VoxTell, and other text-prompted medical segmenters could function as approximate semantic localizers even when they are not accurate enough to be the final segmenter.

These models do not solve the entire problem, but they could become **proposal generators** inside the harness. A rough language-grounded box or mask is often more than enough to derive a reliable point for nnInteractive.

## 14. A custom semantic pointer compiler

We should explicitly separate four stages:

```text
Natural-language intent
        ↓
Semantic grounding
        ↓
Discrete candidate selection
        ↓
Deterministic geometric resolution
        ↓
Typed native-volume coordinate
```

A potential API:

```python
g = ground(
    phrase="the central portion of the infrapatellar fat pad",
    view="sagittal",
    context=["patellar tendon", "femoral condyle", "tibial plateau"],
    return_type="region_candidates",
)

render_marks(g.candidates)

region = g.select(candidate_id=4)

point = anchor(
    region,
    relation="maximally_interior",
    coordinate_space="ras_mm",
)

m1 = seg.point(
    point=point,
    label="positive",
    backend="nninteractive",
)
```

The important output of `ground()` would not just be one coordinate. It should be a `GroundingSet`:

```json
{
  "phrase": "central portion of the infrapatellar fat pad",
  "candidates": [
    {
      "id": 1,
      "region_handle": "region_8dd2",
      "bbox_mm": [12, -32, 41, 38, -8, 72],
      "representative_point_mm": [24, -20, 56],
      "source": "medical_vlm_box",
      "semantic_score": 0.71,
      "interior_clearance_mm": 8.2
    }
  ],
  "ambiguity": 0.34,
  "render_id": "render_marks_024"
}
```

The agent selects a candidate. The harness performs the coordinate conversion.

## 15. “Middle” should be implemented geometrically

“Middle” is not necessarily the centroid.

For a concave object, the centroid can:

- fall outside the structure;
- lie near a narrow boundary;
- produce a poor interactive-segmentation prompt.

A better operational definition is often:

> The point inside the candidate region that maximizes distance to its boundary.

That is the centre of the largest inscribed sphere, obtained from a Euclidean distance transform:

```python
def maximally_interior_point(region_mask, spacing_mm):
    distance = distance_transform_edt(
        region_mask,
        sampling=spacing_mm,
    )
    return np.unravel_index(distance.argmax(), distance.shape)
```

This provides a robust positive point with maximal boundary clearance.

Other phrases can compile into deterministic geometry:

| Phrase | Geometric interpretation |
|---|---|
| “middle of X” | maximal-inscribed point or geodesic centre |
| “superior part of X” | interior point maximizing superior RAS coordinate |
| “anterior margin of X” | anterior boundary point, offset inward |
| “between X and Y” | midpoint between nearest surfaces, validated against tissue |
| “along X” | point or path on the medial axis/skeleton |
| “inside X near Y” | maximize membership in X while minimizing distance to Y |
| “largest component” | component with maximal physical volume |
| “small satellite focus” | non-primary component ranked by semantic and size criteria |
| “brightest part of X” | local intensity maximum constrained to a safe interior region |

The LLM supplies the semantic relation. The runtime computes the exact point.

## 16. Recommended grounding hierarchy

We should use several grounding strategies in descending order of reliability.

### Level 1: Existing anatomical mask or landmark

If a specialist model can segment or detect the IPFP, patellar tendon, femur, or tibia:

```python
ipfp = anatomy.segment("infrapatellar fat pad")
point = anchor(ipfp, "maximally_interior")
```

This is the most reliable.

### Level 2: Text-to-box or text-to-mask model

Use VividMed, VoxTell, SAT, Grounding DINO, or a related model to obtain a rough region:

```python
rough = ground.text_mask("infrapatellar fat pad")
point = anchor(rough, "maximally_interior")
```

The rough mask does not need publication-grade Dice. It only needs to put the point inside the intended structure.

### Level 3: Candidate proposals plus Set-of-Mark

Generate candidates from:

- multiscale superpixels;
- region-growing proposals;
- connected intensity regions;
- generic SAM masks;
- anatomical grid regions;
- local maxima in a phrase-conditioned heatmap.

Render numbered candidates, and let the model select.

### Level 4: Coarse-to-fine spatial search

If no semantic grounding model is available:

1. Divide the image into a coarse grid.
2. Ask the VLM to select a region ID.
3. Crop that region.
4. Subdivide and repeat.
5. Resolve the final cell to a point or local proposal.

This is essentially a visual binary or quadtree search. It converts coordinate generation into repeated classification.

For 3D:

```text
Select axial range → select sagittal/coronal ROI → select local candidate
```

This is slower than a learned grounding head but substantially more reliable than one-shot coordinates.

## 17. Practical MVP additions for the current harness

We do not need to train a new model initially.

### 17.1 `propose_points`

```python
propose_points(
    region=None,
    strategy=[
        "component_centers",
        "maximal_interior",
        "intensity_extrema",
        "uncertainty_peaks",
        "boundary_error",
    ],
    n=9,
)
```

It returns stable IDs and exact native coordinates.

### 17.2 `show_point_candidates`

```python
show_point_candidates(
    candidates,
    views=["axial", "sagittal", "coronal"],
    labels=True,
)
```

The agent sees numbered markers and chooses:

```python
seg.point(candidate_id="P4", label="positive")
```

No raw coordinate ever appears in the model-generated action.

### 17.3 Semantic grounding

```python
ground_region(
    phrase="infrapatellar fat pad",
    mode="box_or_mask",
)
```

Use the resulting region to constrain `propose_points`.

This would directly address the problem encountered in the first experiment without first building a new medical grounding model.

## 18. The most important distinction: selecting versus generating

The harness should make the model do what it is good at:

- identify semantic objects;
- compare alternatives;
- reason about spatial relations;
- choose among candidate actions;
- critique the result.

The harness should do what computers are good at:

- convert grids and coordinate systems;
- find medial points;
- calculate distance transforms;
- select connected components;
- snap to valid voxels;
- project between views;
- rasterize geometry;
- ensure the point lies inside the candidate region.

In other words:

> The model should choose **what** and **where semantically**. The harness should decide **where numerically**.

## 19. Methodological evaluation of semantic-to-coordinate grounding

Add a localization experiment separate from final segmentation.

### Conditions

1. Free-text coordinate emission
2. Screenshot click
3. Coarse-to-fine grid selection
4. Set-of-Mark point selection
5. Text-grounded box → deterministic interior point
6. Text-grounded mask → deterministic interior point
7. Oracle interior point

### Localization metrics

- Point-inside-target rate
- Distance to nearest target voxel in millimetres
- Distance to target medial axis
- Boundary clearance
- Success@K candidates
- Left/right and anatomical-relation accuracy
- Time, tokens, and model calls
- Downstream Dice after exactly one nnInteractive prompt

That last metric is especially strong:

> How much downstream segmentation quality is lost solely because of the language-to-coordinate interface?

We may find that the LLM's semantic reasoning is substantially better than the original experiment suggested and that most of the apparent failure came from coordinate serialization.

## 20. Decomposing total error

We could formalize the pipeline as:

\[
\text{final error}
=
\text{semantic grounding error}
+
\text{coordinate resolution error}
+
\text{segmentation backend error}
+
\text{refinement/stopping error}.
\]

The original system entangled all four. A semantic pointer tool would let us measure them separately.

That could itself be a significant methodological contribution:

> **Agentic segmentation systems should treat semantic intent grounding and executable coordinate resolution as separate modules.**

The closest precedents are coordinate-free GUI grounding and Set-of-Mark prompting, but the adaptation to true 3D medical-image interaction—especially phrase → candidate region → physically typed point → volumetric segmentation—still appears quite open.

## 21. Additional references for grounding and coordinate resolution

1. Wu Q et al. [GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents](https://arxiv.org/abs/2506.03143).
2. Yang J et al. [Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V](https://arxiv.org/abs/2310.11441); [code](https://github.com/microsoft/SoM).
3. Deitke M et al. [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](https://arxiv.org/abs/2409.17146).
4. Zhao B et al. [M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models](https://arxiv.org/abs/2404.00578); [code](https://github.com/BAAI-DCAI/M3D).
5. Luo L et al. [VividMed: Vision Language Model with Versatile Visual Grounding for Medicine](https://aclanthology.org/2025.naacl-long.89/); [code](https://github.com/function2-llx/MMMM).
6. [MedGround-R1: Advancing Medical Image Grounding via Spatial-Semantic Rewarded GRPO](https://github.com/bio-mlhui/MedGround-R1).

