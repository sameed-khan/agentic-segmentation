# Optimization and refinement literature review (for the agentic 3D segmentation project)

**Scope.** Literature review for the harness described in `research/experiment_log.md` (sections 0, 3, 5): an LLM
agent (pi framework, Claude Sonnet 5.5, subscription access, no fine-tuning) that segments soft-tissue tumours in
3D MRI (MSTT-199, 30-case development set) by driving nnInteractive with point / box / scribble / lasso prompts
chosen via set-of-marks (SoM) pointing.

**Questions answered.** (1) prompt/program optimization without weight updates; (2) why agents under-use tools and
terminate early, and fixes without fine-tuning; (3) agentic / LLM-driven interactive segmentation; (4) refinement
tools that reduce the precision of pointing; (5) small/subtle lesion localization by VLMs.

**How this review was produced (verification policy).**
- Search: `pi-web-access` MCP tools (`web_search` via Exa/DuckDuckGo/degoog, `fetch_content`), plus the arXiv API
  and the public OpenAlex API for metadata. Date of search: **2026-10-19** (papers up to arXiv 2610 are included).
- **✅ = primary source fetched in this session** (arXiv abstract page, arXiv HTML, journal/blog page, or full PDF
  text extracted with `pypdf`); numbers quoted from that source. **⚠️ = metadata verified via the OpenAlex DOI
  record (title/year/venue) but the publisher full text is paywalled and could not be fetched**; claims from those
  items are limited to their indexed abstract. **❓ = no verified source found** (explicitly flagged as a gap).
- Nothing in this document is inferred or invented. Where a number is quoted it is tagged with its source; where
  a claim about our own runs is made it points to `research/experiment_log.md` (E-numbers).
- Cost estimates use our measured run costs (E9/E11: medium ≈ $0.04/case, xhigh ≈ $0.25/case API-equivalent; we
  pay subscription + wall-clock, not per-token dollars, so "cost" below is mostly **quota and wall-clock**).

---

## 0. Executive summary — 10 recommendations, ranked by expected value / cost

1. **Force at least one corrective nnInteractive prompt per case, with the correction target proposed from the
   current mask's error-prone regions (highest EV, cheap).** nnInteractive is *trained* on exactly this loop: it
   simulates follow-up prompts sampled from false-positive/false-negative connected components of the current
   prediction (size-proportional, Perlin-fragmented, previous prompts decayed ×0.9; [nnInteractive,
   arXiv:2503.08373](#c-nnint)). Our agent issues 1.0–1.7 prompts/case (E5–E9, E11); the interactive baseline on
   the same benchmark improves +0.22 category-macro Dice from 1→5 points and +0.04 more after 5 corrective clicks
   ([SAMI3D-DW, arXiv:2609.25743](#c-sami3d)). LeCor shows corrective rounds + test-time adaptation moving Dice
   0.298→0.757→0.827 on lung-tumour CT ([arXiv:2609.09477](#c-lecor)). Harness cost: 1–2 extra tool calls/case
   (~+20–50% wall-clock), ~0 API cost; highest probability of breaking the "accept-first-mask" plateau.
2. **Make under-segmentation mechanically visible: quantitative QC in the review step (low cost, high EV).**
   Our errors are precision≈1.0 / recall 0.3–0.6 with masks stopping at cranio-caudal ends (E6, E8). A
   deterministic QC block (component count; mask volume vs. intensity-similar neighbour growth; "does the mask
   end where the tumour signal ends?") plus an explicit checklist can trigger the loop from #1. Precedent: a
   VLM judge for mask quality reaches r=0.902 with Dice and 0.06 s per 3D mask ([SegAE,
   arXiv:2601.14406](#c-segae)); agent-as-judge and LLM-critic literature shows *external* checks catch errors
   self-review misses ([Agent-as-a-Judge 2410.10934](#c-aaj), [CriticGPT 2407.00215](#c-criticgpt)).
3. **Add a zoom/crop tool for small, subtle, or low-contrast targets (medium cost, high EV on the 4–7/30 misses).**
   V* (guided visual search; [2312.14135](#c-vstar)), ZoomEye (tree search over zoomed regions;
   [2411.16044](#c-zoomeye)), Chain-of-Focus (zoom+RL; +5% on the V* benchmark across resolutions 224–4K;
   [2505.15436](#c-cof)) and DeepEyes ("thinking with images" via cropping; [2505.14362](#c-deepeyes)) all show
   that crop-zoom at test time recovers fine detail that a full-slice render destroys. DeepTumorVQA names small
   tumour detection as a leading failure of medical VLMs ([2505.18915](#c-deeptumorvqa)). Our persistent misses
   F15 (2 mL), L33 (T2 fat-suppressed), N40 are exactly this regime (E10–E12). Cost: render a crop around a mark
   at higher effective resolution; ~1 extra view per hard case.
4. **Prefer two-plane "extent seeding" over a single point (very low cost, medium-high EV).** nnInteractive's
   expert-scribble benchmark gets 84.3 Dice from scribbles on only **3 slices** vs. all-slice scribbles
   (87.3) and beats a 2D competitor by ~7 Dice ([nnInteractive](#c-nnint), Table 1). Our under-segmentation is
   concentrated at z-ends (E6); asking for one scribble/point on the first and last visible tumour slice is a
   prompt-level change the existing tools already support.
5. **Budget forcing + an external verifier pass before `seg_finish` (low cost, medium EV).** s1 shows simply
   appending "Wait" to lengthen thinking improves competition math (+7 points AIME24 extrapolation) without
   fine-tuning ([2501.19393](#c-s1)); underthinking work shows models switch thoughts prematurely and a
   thought-switching penalty helps ([2501.18585](#c-underthink)). Caveat: a systematic study of test-time scaling
   for *agents* finds neither sequential (more turns) nor parallel (more trajectories) scaling reliably helps
   without a verifier — the "verification gap" ([2602.18998](#c-ttsagents)). So pair forcing with #2's verifier,
   not with more turns alone. Cost: one extra model turn (~+15–30% wall-clock).
6. **Run GEPA on the system prompt with a stratified 15/15 split and 3 folds — after #1–#5 fix the loop
   (medium EV, medium cost).** GEPA beats GRPO by up to 20% with up to 35× fewer rollouts and beats MIPROv2 by
   >10% (e.g., +12% AIME-2025) on six compound AI systems, works with closed models (GPT-4.1 Mini), and has strong
   cross-model prompt transfer (+9% when optimising on Qwen3-8B and testing on GPT-4.1 Mini)
   ([GEPA 2507.19457](#c-gepa)). GEPA's own tasks use 45–150 training / 45–300 validation examples; with 30 dev
   cases we must hold out and stratify (6/histotype). Budget: ~50–150 candidate evaluations × minibatch of 5 =
   250–750 case-runs (~$60–190 API-equivalent at xhigh) plus reflection calls; likely multi-day under subscription
   rate limits. Risk: prompt overfitting to 30 cases — mitigate with held-out validation, Pareto selection, and a
   freeze before the 169-case test split.
7. **Add an ACE-style evolving "failure playbook" as agent context (medium EV, low risk).** ACE treats context as
   an evolving playbook with generation/reflection/curation and reports +10.6% on agent benchmarks, +8.6% on
   finance, with −86.9% adaptation latency and ~75% fewer rollouts than GEPA in its comparison, adapting from
   execution feedback without labels ([ACE 2510.04618](#c-ace)). It is inspectable, so it is a safer first
   "learning" mechanism than a black-box prompt search. Risk: context bloat/collapse if curation is weak.
8. **Expose a region-grow / adaptive-brush tool only after the loop is fixed (medium EV, higher implementation
   cost).** 3D Slicer "Grow from seeds" (improved GrowCut) and "Fill between slices" (slice interpolation), plus
   ITK-SNAP's adaptive paintbrush, are the classic tools that turn a rough pointer into a full structure; ITK-SNAP
   4.4.0 (Sep 2025) now integrates nnInteractive itself for scribble/polygon→3D AI segmentation
   ([Slicer docs](#c-slicerdocs), [ITK-SNAP DLS docs](#c-itksnapdls), [GrowCut ref. in Slicer docs](#c-growcut)).
   Human-efficiency precedent: nnInteractive user study 179±114 s vs 635±343 s manual (−72%)
   ([nnInteractive](#c-nnint)); MedSAM2 reports >85% manual cost reduction in a 5,000-CT-lesion user study
   ([2504.03600](#c-medsam2)). Implement cheaply as "local threshold + connected component grow within a box" and
   compare against plain nnInteractive prompts.
9. **Ablate set-of-marks density and rendering on the 3 persistent misses (low cost, targeted).** No published
   study of SoM mark-density effects on VLM pointing was found in this search (❓ gap), but SoM-style marking is now
   used inside segmentation agents (Seg-Agent renders candidate regions as SoM for interactive reasoning;
   [2605.12953](#c-segagent2)). Our Phase 2 plateau (S≈0.90 regardless of histotype/description/example, E10)
   suggests the ceiling is not prompt knowledge but visual access; a density/zoom ablation on F14/F15/L33 is a
   cheap, decisive test.
10. **Keep evaluation discipline: 15/15 dev split, paired bootstrap CIs, pre-registered gates.** Run-to-run
    difference in our data is ±0.03–0.05 Dice for a 30-case mean (E9), so any change must be judged with paired
    tests/CIs ([Adding Error Bars to Evals 2411.00640](#c-errorbars)); GEPA itself notes that validation rollouts
    dominate its budget and recommends smaller validation subsets ([GEPA](#c-gepa)).

**Overall reading of the evidence.** Nothing in the literature suggests the agent's core problem is *reasoning*;
it matches known failure modes of (a) under-use of iterative refinement tools that are trained for exactly this
error type, (b) loss of small/low-contrast detail in the rendered observation, and (c) premature termination
without external verification. The highest-EV fixes are harness/tool/prompt-level, not model-level; prompt
optimization (GEPA/ACE) is worth doing only after the loop and the observation channel are fixed.

---

## 1. Prompt / program optimization for LLM agents without weight updates

| Work (year, venue) | Link | What is optimised / feedback | Key verified numbers | Relevance to us |
|---|---|---|---|---|
| **GEPA** (2025→2026, ICLR 2026 Oral) | [2507.19457](https://arxiv.org/abs/2507.19457) ✅ abs+PDF | Every prompt/module of a compound AI system; natural-language reflection over sampled trajectories (reasoning, tool calls, tool outputs) + Pareto evolutionary search; no weights | Outperforms GRPO by 6% avg / up to 20% with up to 35× fewer rollouts; >10% over MIPROv2 (+12% AIME-2025); 4–35× fewer rollouts to optimal; train-only budget 79–737 rollouts; works on closed GPT-4.1 Mini; +9% cross-model transfer; uses 45–150 train / 45–300 val / test splits | The best fit for our setting: subscription API, no weights, executable per-case metric (Dice) available offline, tool-call traces already logged. Risk: val rollouts dominate cost; 30 cases is at the low end (hence 15/15 + folds) |
| **DSPy** (2023, preprint) | [2310.03714](https://arxiv.org/abs/2310.03714) ✅ PDF | Declarative LM programs; compiler bootstraps few-shot demos / finetunes; metric-driven | Pipelines outperform standard few-shot prompting by >25% (GSM8K) / 65% (HotPotQA); explains program-level optimisation | Framework/model for "prompts as parameters"; MIPROv2/SIMBA/GEPA live in it |
| **MIPROv2** (2024, EMNLP 2024) | [2406.11695](https://arxiv.org/abs/2406.11695) ✅ abs+PDF | Instructions + few-shot demos for *all* modules of a multi-stage program; stochastic mini-batch evaluation + Bayesian surrogate over proposal hyper-parameters (meta-optimisation); no module-level labels | Up to +13% accuracy; wins 5/7 multi-stage programs vs. baselines; 300+ minibatch trials in one reported optimisation | The direct GEPA baseline; its minibatch strategy is the standard answer to scarce eval budget — relevant for our 30 cases |
| **SIMBA** (DSPy optimizer) | [DSPy docs](https://dspy.ai/api/optimizers/SIMBA/) ✅; comparison paper [2602.17937](https://arxiv.org/abs/2602.17937) ✅ | Instructions (and demos) of a program; LLM introspects its own mini-batches, adds rules for high-variance examples or successful demos; no separate paper found (❓ **no arXiv paper**; documented as a DSPy optimizer; a community blog describes it as GEPA's predecessor) | The controlled comparison ([2602.17937](https://arxiv.org/abs/2602.17937)) found MIPROv2 most stable for CoT, **SIMBA largest gains for ReAct tool agents** (especially larger models), avoiding unnecessary tool calls | Evidence that reflective-rule optimisers can change *tool-call behavior* (ReAct), which is our exact failure mode; no standalone paper ⇒ cite as DSPy component |
| **TextGrad** (2024, preprint) | [2406.07496](https://arxiv.org/abs/2406.07496) ✅ abs+PDF | Textual "gradients" back-propagated through arbitrary computation graphs (prompts, code, molecules); LLM-generated natural-language critiques; test-time or training-time | GPQA zero-shot 51%→55%; +20% relative on LeetCode-Hard; radiotherapy-planning and molecule-design case studies | Generalises to any textual parameter, but less sample-efficient/structured than GEPA for multi-module agents; useful conceptual framing of "textual critique as gradient" |
| **OPRO** (2023, ICLR 2024) | [2309.03409](https://arxiv.org/abs/2309.03409) ✅ abs | A single instruction prompt; meta-prompt containing previous candidates + scores, LLM proposes next | +8% GSM8K / +50% BigBench-Hard over human prompts | Simple baseline; no traces/reflection, poor fit for tool-use traces |
| **Promptbreeder** (2023, preprint) | [2309.16797](https://arxiv.org/abs/2309.16797) ✅ PDF | Task prompts **and** mutation prompts (self-referential evolution); fitness from task accuracy | Outperforms CoT and Plan-and-Solve on arithmetic/commonsense; evolves hate-speech prompts | Prompt-only (no tool traces); not evaluated on agents; mostly historical |
| **ACE** (2025→2026, ICLR 2026) | [2510.04618](https://arxiv.org/abs/2510.04618) ✅ abs+PDF | Evolving "context playbook" (offline system prompt / online memory); generator→reflector→curator incremental delta updates; execution feedback, can work without labels | +10.6% on agent benchmarks, +8.6% on finance; −86.9% adaptation latency avg; fewer rollouts than GEPA (e.g., −75.1% on one benchmark); matches top AppWorld agent with smaller model | A safer, inspectable alternative to prompt search for turning repeated failures (F15/L33/N40) into context lessons; risk of context collapse is addressed by curation |
| **Trace / OptoPrime** (2024, NeurIPS 2024) | [2406.16218](https://arxiv.org/abs/2406.16218) ✅ abs+PDF | Any parameters of a workflow (prompts, code, hyper-params, robot controllers) via OPTO: execution trace + rich feedback as "gradient"; first-order numerical optimisation supported | OptoPrime competitive with specialised optimisers; reported ~10% higher accuracy on BigBench-Hard than domain baselines | Shows traces+feedback are a general optimisation interface; the library is heavier than GEPA for a single-prompt harness |
| **AFlow** (2024→2025, ICLR 2025) | [2410.10762](https://arxiv.org/abs/2410.10762) ✅ PDF | Workflow topology (code-represented agent graphs) via MCTS; execution feedback | +5.7% avg over state-of-the-art workflows on 6 benchmarks; 4.55% of their inference cost on specific tasks | Relevant only if we let the harness's control flow (not just prompt) evolve; our loop is small — prompts first |
| **ADAS / Meta Agent Search** (2024→2025, ICLR 2025) | [2408.08435](https://arxiv.org/abs/2408.08435) ✅ PDF | Inventing new agent architectures in code, archive-driven; task feedback | +13.6 F1 on DROP; +14.4% MGSM; discovered agents transfer (+25.9%/+13.2% on GSM-Hard); agents maintained across domains | Long-horizon, expensive; not competing with a single-prompt fix |
| **GEPA follow-ups (2025–2026)** | [Adaptive-GEPA 2609.38762](https://arxiv.org/abs/2609.38762) ✅; [Ranking-PE 2609.40361](https://arxiv.org/abs/2609.40361) ✅; [CMVF 2607.24354](https://arxiv.org/abs/2607.24354) ✅; [GEPA-RoB 2512.01452](https://arxiv.org/abs/2512.01452) ✅ | Router + specialists (Adaptive-GEPA); AUROC-aligned Pareto rows for clinical MLLMs (Ranking-PE); *visual* failure diagnosis fed to the optimiser (CMVF); GEPA applied to RCT risk-of-bias assessment | Adaptive-GEPA family-mean 52.6→70.6 vs 62.5 GEPA / 54.0 GRPO at 18k scored calls; CMVF +2.4 points avg (up to +6.5) over strongest APO baseline on 12 VQA datasets / 4 VLMs and transfers as text checklists; GEPA-RoB best in several RoB domains (+30–40% in Random Sequence Generation) | Direct evidence that (a) GEPA scales to multimodality and (b) **visual feedback matters**: our optimiser should see the rendered images/diagnoses, not just the Dice scalar (CMVF) |
| **APO applied to medical VLMs** | [Prompt Triage 2511.11898](https://arxiv.org/abs/2511.11898) ✅ (DSPy, 5 medical tasks, 10 VLMs) | Structured prompt optimisation (DSPy) for medical vision-language pipelines, weight-agnostic | Median +53% relative over zero-shot; +300–3400% where zero-shot is very low | Closest published "APO for medical VLMs" result; no segmentation task, no agent loop |
| **APO benchmark / small-data rigor** | [Chess APO benchmark 2610.00416](https://arxiv.org/abs/2610.00416) ✅; [Tabular fact verification 2602.17937](https://arxiv.org/abs/2602.17937) ✅; [Error bars 2411.00640](https://arxiv.org/abs/2411.00640) ✅ | Benchmarking/sample efficiency of APO on frozen LLMs; statistical guidance for evals | Chess: 1,118 puzzles, six APO algorithms, eight models, tests transfer; tabular paper: three DSPy optimisers × 4 prompting techniques × 3 model families; error-bars paper: formulas/planning for eval CIs | Supplies the methodology for a defensible 30-case study: cheap deterministic scoring, held-out splits, paired CIs |

**What is *not* in the literature (verified absence).** Despite targeted searching (GEPA/DSPy/TextGrad/OPRO/ACE
+ “segmentation/medical imaging/agent”), **no published application of these optimisers to an interactive 3D
segmentation agent was found**; the nearest published applications are medical VLM benchmarks (Prompt Triage),
RCT risk-of-bias classification (GEPA-RoB), clinical AUROC ranking (Ranking-PE) and VQA (CMVF). Our harness would
be an early application to a tool-using *segmentation* agent.

**Sample-efficiency and overfitting with ~30 cases.** GEPA's six tasks use 45–150 train and 45–300 validation
examples and it warns that most rollout budget goes to validation (it suggests smaller validation subsets). MIPROv2
uses stochastic minibatching precisely to avoid spending the whole budget on large validation sets. With 30 cases
(6 per histotype) the practical design is: stratified 15 train / 15 validation (optionally 3-fold), batch size
5–10, treat per-case Dice as the feedback, and keep the prompt frozen once chosen (no further peeking). Report the
final prompt only on the untouched 169-case test split → the development set is already "burned" in E1–E12.

---

## 2. Why agents under-use tools / terminate early, and fixes without fine-tuning

| Work (year, venue) | Link | Mechanism / finding | Key verified numbers | Relevance to us |
|---|---|---|---|---|
| **Reflexion** (2023, NeurIPS 2023) | [2303.11366](https://arxiv.org/abs/2303.11366) ✅ abs | Verbal self-reflection stored in episodic memory across trials; binary/scalar feedback converted to text; no weight updates | HumanEval 91% pass@1 vs 80% for prior GPT-4 SOTA | Across-case learning mechanism; per-case (single-run) it only helps if the agent retries within budget |
| **Self-Refine** (2023, NeurIPS 2023) | [2303.17651](https://arxiv.org/abs/2303.17651) ✅ abs | Same LLM generates → critiques → refines its own output iteratively | ~20% absolute average improvement over one-step generation across 7 tasks (GPT-3.5/4) | The generic "review then edit" pattern our two-stage `seg_finish` implements; gains depend on feedback richness — our review is visual, which is harder |
| **CRITIC** (2023, ICLR 2024) | [2305.11738](https://arxiv.org/abs/2305.11738) ✅ abs | Self-correct only *with external tools* (search, code interpreter) validating the output, then revise | Consistent gains on QA, program synthesis, toxicity; emphasises external feedback as essential | Supports adding a tool-verifier (QC metrics, VLM judge) rather than relying on the agent's own opinion |
| **LLMs cannot self-correct reasoning yet** (2023, ICLR 2024) | [2310.01798](https://arxiv.org/abs/2310.01798) ✅ abs | Intrinsic self-correction (no external feedback) often fails or degrades reasoning performance | Critique of self-correction without external signal | Direct warning for our "confirm without editing" behaviour: the fix must change the *evidence* the model sees |
| **Chain-of-Verification** (2023, ACL Findings 2024) | [2309.11495](https://arxiv.org/abs/2309.11495) ✅ abs | Draft → plan verification questions → answer them independently → final verified response | Reduces hallucination across list QA, MultiSpanQA, longform | Template for a checklist verification pass (e.g., “is the mask’s z-extent consistent with signal?”) |
| **s1 / budget forcing** (2025) | [2501.19393](https://arxiv.org/abs/2501.19393) ✅ abs | Append "Wait" to lengthen thinking / truncate to shorten; controls test-time compute after SFT; s1-32B | Exceeds o1-preview by up to 27% on competition math; 50%→57% on AIME24 with forcing | Inference-time forcing analogue for agents: require a second prompt round/pass before finishing; cheap to test |
| **Underthinking** (2025, ICLR 2025) | [2501.18585](https://arxiv.org/abs/2501.18585) ✅ abs | o1-like models switch thoughts prematurely; thought-switching penalty (TIP) at decoding | Frequent thought switching correlates with wrong answers; TIP improves accuracy without fine-tuning | Explains "one click then finish"; decoding-level fix is unavailable in pi, but prompt-level penalties/requirements are |
| **OptimalThinkingBench** (2025) | [2508.13141](https://arxiv.org/abs/2508.13141) ✅ abs | Joint over/under-thinking benchmark; no model thinks optimally | Thinking models overthink simple queries; non-thinking models underthink hard ones | Perspective on reasoning-effort knobs: our xhigh gains were *search* gains, not refinement gains (E11) |
| **Test-time scaling of general LLM agents** (2026) | [2602.18998](https://arxiv.org/abs/2602.18998) ✅ abs | Systematic sequential (more turns) vs parallel (trajectory sampling) scaling in a realistic benchmark | Neither axis consistently yields gains; two causes: scaling plateau and **verification gap** | Strong caution: simply giving the agent more turns/budget will not fix refinement unless a verifier exists |
| **Let's Verify Step by Step** (2023, preprint) | [2305.20050](https://arxiv.org/abs/2305.20050) ✅ abs | Process reward models (step-level supervision); active learning for labels | 78% of MATH test subset solved by process-supervised model | Blueprint for a per-step reward/verifier; training a PRM is out of scope for us, but a heuristic step-checker is not |
| **Agent-as-a-Judge** (2024) | [2410.10934](https://arxiv.org/abs/2410.10934) ✅ abs | Agentic evaluation of agents; intermediate step feedback instead of final outcome only | Outperforms LLM-as-a-Judge and is as reliable as human evaluation on DevAI (55 tasks, 365 requirements) | Supports using an agentic verifier on the agent's own transcript/renders (budget permitting) |
| **CriticGPT** (2024) | [2407.00215](https://arxiv.org/abs/2407.00215) ✅ abs | Trained LLM critics write natural-language critiques of code for human reviewers | Critics preferred over human critiques in 63% of cases; catch more bugs; but hallucinate bugs | Evidence that an *external* critic (even fallible) catches errors; warns that critic hallucinations can mislead |
| **SegAE** (2026, ISBI 2026) | [2601.14406](https://arxiv.org/abs/2601.14406) ✅ abs | Lightweight VLM predicts label quality across 142 structures; synthetic training data | r=0.902 with GT Dice; 0.06 s per 3D mask; reduces quality-check time by 70% and annotation cost by 1/3 | Direct precedent for a cheap mask-quality verifier on rendered/segmented data; could trigger corrective prompts |
| **MOTIVE** (2026) | [2610.07018](https://arxiv.org/abs/2610.07018) ✅ abs | Multi-view self-verification with reliability-guided rethinking for VLMs | Verification performance is highly sensitive to prompt/criterion; strong verifiers needed; no single prompt dominates | If we self-verify with the same model, expect unstable judgments; use multiple views/criteria and an accept-or-rethink rule |
| **Model-adaptive tool necessity / knowing–doing gap** (2026) | [2605.14038](https://arxiv.org/abs/2605.14038) ✅ abs | Compares when a tool is *necessary* with when the model actually calls it | Mismatch 26.5–54.0% (arithmetic) and 30.8–41.8% (factual QA); two stages: cognition (believes necessary) vs execution (actually calls) | The best verified quantification of "the model knows it should call a tool but doesn't" — the same gap as our agents accepting the first mask |
| **Tunable tool-call rates** (2026) | [2608.25198](https://arxiv.org/abs/2608.25198) ✅ abs | Steering a single residual-stream direction moves call rate 0→>90% with no prompt change | Steering works both ways and generalises to unseen tools; traces a cost/accuracy Pareto frontier | Not applicable (no weight/activations access) but confirms tool-call rate is a controllable policy axis, not a knowledge gap |
| **MAST: why multi-agent systems fail** (2025) | [2503.13657](https://arxiv.org/abs/2503.13657) ✅ abs | Taxonomy of 14 failure modes over 1,600+ traces in 7 MAS frameworks | Three clusters incl. **task verification**; LLM-as-judge pipeline agrees with humans | Independent confirmation that verification failures are a primary agent failure category |
| **Cross-modal visual feedback (CMVF)** (2026) | [2607.24354](https://arxiv.org/abs/2607.24354) ✅ abs | APO optimiser *sees failed images* and extracts visual blind-spot patterns into text checklists | +2.4 points avg (up to +6.5) on 12 VQA datasets / 4 VLMs; checklists transfer across models | Strong argument to give our prompt optimiser/verifier the render, not just Dice; checklist output is cheap context |

**Synthesis for our failure mode.** The literature distinguishes (i) *knowledge* gaps ("the model can't do it"),
(ii) *execution* gaps (knows a tool would help but doesn't call it; 26–54% measured mismatch), and (iii)
*verification* gaps (no reliable signal to say the output is wrong). Our E5–E11 pattern — one prompt, no edits,
review confirms without changing anything, and xhigh only improves search — is (ii)+(iii), not (i). Without
fine-tuning, the available levers are: force the call (budget forcing / hard rules), improve the evidence (QC
blocks, zooms, renders), and add an external verifier (QC metrics/VLM judge). More turns alone is explicitly
contraindicated by [2602.18998](https://arxiv.org/abs/2602.18998).

---

## 3. Agentic / LLM-driven interactive segmentation

| Work (year, venue) | Link | What it does | Key verified numbers | Relevance to us |
|---|---|---|---|---|
| **SegAgent / HLMAT** (2025, CVPR 2025) | [2503.08625](https://arxiv.org/abs/2503.08625) ✅ abs+PDF | MLLM imitates human annotator trajectories: text-based click points iteratively refine a mask; multi-step MDP; PRM predicts mIoU; supports refinement/annotation filtering | Competitive IoU vs. SOTA (except GLaMM) on RES datasets; refinement over box baseline on all datasets; **fine-tuned on human trajectories** | Closest published "MLLM as annotator with interactive clicks"; confirms (a) click generation is learnable, (b) refinement is where they add value. We cannot fine-tune — GEPA/prompting is our substitute |
| **Seg-Zero** (2025) | [2503.06520](https://arxiv.org/abs/2503.06520) ✅ abs | Decoupled reasoner + segmentation model; RL (GRPO) with format+accuracy rewards; explicit CoT; no reasoning data | 57.5 zero-shot on ReasonSeg, +18% over LISA-7B; emergent test-time reasoning | Shows RL-trained agents beat SFT on reasoning segmentation; not applicable without weights, but its reward design (format + accuracy) is a template for our QC signals |
| **Seg-R1** (2025) | [2506.22624](https://arxiv.org/abs/2506.22624) ✅ abs | LMM emits point+box prompts for SAM2; pure RL (GRPO); no text supervision | 0.873 S-measure COD10K; 71.4 cIoU RefCOCOg zero-shot; 56.7 gIoU ReasonSeg | Prompt-generation policy is RL-learnable; again evidence that iterative/prompted segmentation can be strong but requires training we can't do |
| **SegLLM** (2024) | [2410.18923](https://arxiv.org/abs/2410.18923) ✅ abs | Multi-round interactive *reasoning* segmentation with conversational memory (previous masks fed back) | >20% over prior multi-round methods on MRSeg; +5.5 cIoU / +4.5 Acc@0.5 on referring tasks | Memory of previous masks is useful; our harness already feeds the last mask back into renders |
| **LISA** (2024, CVPR 2024) | [2308.00692](https://arxiv.org/abs/2308.00692) ✅ abs | Reasoning segmentation via `<SEG>` token; the baseline that Seg-Zero/Seg-R1 improve on | >1k reasoning-segmentation samples; robust zero-shot; 239 samples fine-tune helps | Background for "LLM-embedded segmentation"; not interactive-click based |
| **VoxelPrompt** (2024→2025) | [2410.08397](https://arxiv.org/abs/2410.08397) ✅ abs+PDF | End-to-end 3D medical agent: LLM generates executable code that calls a jointly trained vision network; open-language radiology tasks | Accuracy similar to specialist single-task models; hundreds of anatomical/pathological features; compositional workflows | Best existing "LLM agent for 3D medical image analysis", but it drives a *trained* vision network by code, not an interactive prompter |
| **MedRAX** (2025) | [2502.02673](https://arxiv.org/abs/2502.02673) ✅ abs | Tool-using agent for chest X-ray with a benchmark of complex queries | ChestAgentBench: 2,500 queries / 7 categories; SOTA vs open + proprietary models; no extra training | Template for tool-orchestration evaluation; 2D CXR, no interactive refinement |
| **MedSAM-3 / MedSAM-3 Agent** (2025) | [2511.19046](https://arxiv.org/abs/2511.19046) ✅ abs | Text-promptable SAM3-based medical segmenter + MLLM agent-in-the-loop for iterative refinement | SOTA vs specialists across X-ray/MRI/US/CT/video; agent performs reasoning + iterative refinement | Directly comparable architecture (agent + promptable medical seg model); no agent benchmark numbers in abstract |
| **MedSAM-Agent** (2026) | [2602.03320](https://arxiv.org/abs/2602.03320) ✅ abs+PDF | Interactive medical segmentation as multi-step decision process; hybrid prompting for expert-curated trajectories; two-stage RLVR + clinical-fidelity process reward to promote interaction parsimony; ΔIoU-based trajectory filtering | SOTA across 6 modalities / 21 datasets; explicitly addresses redundant actions and single-turn rigidity | The clearest statement that *multi-turn* interaction policy is the bottleneck and that process rewards fix it; without RL we can only shape this via prompts/rules |
| **MediRound** (2026, LNCS) | [10.1007/978-3-032-37196-6_21](https://doi.org/10.1007/978-3-032-37196-6_21) ⚠️ | Multi-round entity-level reasoning segmentation in medical images | Metadata verified (LNCS 2026); abstract not fetched | Shows the medical field is moving to multi-round; paywalled → not used for numbers |
| **MedSAM2** (2025) | [2504.03600](https://arxiv.org/abs/2504.03600) ✅ abs+PDF | Promptable 3D medical/video segmentation (SAM2 fine-tune); large human-in-the-loop study | 455k 3D pairs + 76k frames; user study: 5,000 CT lesions, 3,984 liver-MRI lesions, 251,550 echo frames; >85% manual cost reduction | Human-efficiency anchor for refinement tools; supports iterative correction as the intended workflow |
| **SAM / SAM 2** (2023 / 2024) | [2304.02643](https://arxiv.org/abs/2304.02643) ✅; [2408.00714](https://arxiv.org/abs/2408.00714) ✅ | Promptable 2D/3D segmentation foundation models; data engines with iterative refinement | SAM: 1B masks / 11M images; SAM2: 3× fewer interactions in video, 6× faster than SAM in images | Underlying promptable paradigm; our model (nnInteractive) is the 3D medical adaptation |
| **nnInteractive** (2025) | [2503.08373](https://arxiv.org/abs/2503.08373) ✅ abs+PDF | 3D promptable open-set segmentation with points/boxes/scribbles/**lasso**; auto-zoom; trained on error-region corrective interactions with simulated user agents | 120+ datasets; point +10.1 Dice over ScribblePrompt, box +14.9 over SAM2, scribble +23.8; interactive points reach >70 Dice; lasso best AUC 83.42; expert scribbles on 3 slices: 84.3; user study 179±114 s vs 635±343 s manual (−72%); median Dice vs specialist 0.794 vs inter-expert 0.842 | **Our tool.** Its design already assumes refinement prompts; our agent under-uses it. Its lasso result means a coarse 2D outline beats a point when extent is the problem |
| **SlicerNNInteractive** (2025) | [2504.07991](https://arxiv.org/abs/2504.07991) ✅ abs | Client–server nnInteractive integration into 3D Slicer; all four prompt types | 5 pages; decouples inference from UI; keyboard shortcuts | Shows tool exposure of nnInteractive inside an established viewer — an alternative wrapper for our agent if tool-call friction is a problem |
| **ITK-SNAP 4.4.0 DLS integration** (2025) | [ITK-SNAP DLS docs](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html) ✅ page | ITK-SNAP 4.4 (Sep 2025) integrates nnInteractive: scribbles (paintbrush) or polygon outlines are transformed into AI 3D segmentation | Qualitative (docs) | Confirms the "human draws roughly → AI finishes" workflow is now standard in open tools; matches our planned tool set |
| **SAMI3D-DW** (2026) | [2609.25743](https://arxiv.org/abs/2609.25743) ✅ abs | Interactive 3D segmenter evaluated under simulated user interactions on a large CT/MR benchmark | 4,326 cases / 219 datasets / 107 categories; **category-macro Dice: 1 point 0.5756 (nnInteractive 0.5316) → 5 points 0.7771 (0.7495); bbox 0.7129 (0.6530); after 5 corrective clicks 0.8004 (0.7868)** | The best current number for “what corrective clicks buy” on nnInteractive-like models: +0.22 from 1→5 points, +0.04 from corrective rounds on top |
| **SLIP** (2026) | [2607.22332](https://arxiv.org/abs/2607.22332) ✅ abs | Low-latency interactive 3D prompting with decoupled encoding + patch memory; reversible prompts; real-user study | Identifies nnInteractive's latency and irreversible prompting as annotation-workflow limits; controlled prospective user study: 3 clinical annotation tasks, 6 expert participants (abstract) | If we add refine loops, latency per interaction matters (nnInteractive 0.3–0.8 s on our A10, E1; more in Colab) |
| **LeCor** (2026) | [2609.09477](https://arxiv.org/abs/2609.09477) ✅ abs | Treats each clinician correction as a training signal; meta-learned case adapters (test-time training) on SAM3 for lung-tumour CT | Single point 0.298 → 0.757 after fine-tune; 7 correction rounds → 0.765 (memory conditioning alone stops improving after 6 rounds); LeCor 0.787→0.827 after 7 rounds; cases never reaching 0.80: 47→27; 3 rounds ≈ fine-tuned 7 | Shows corrective clicks have *diminishing* returns under pure memory conditioning and that adaptation is needed beyond ~6 rounds; relevant to how many refine rounds are worth budgeting |
| **Dynamic Prompt Generation for Interactive 3D Seg. Training** (2025) | [2510.03189](https://arxiv.org/abs/2510.03189) ✅ abs | Training strategy with dynamic volumetric prompt generation + adaptive cropping; entry to the MICCAI challenge on Foundation Models for Interactive 3D Biomedical Image Segmentation | Average final Dice 0.6385, NSD 0.6614, AUC-Dice 2.4799 on the challenge | Confirms an active community benchmark on exactly our task class; useful reference point for our ~0.7 mean |
| **Seg-Agent (training-free, SoM)** (2026) | [2605.12953](https://arxiv.org/abs/2605.12953) ✅ abs | Explicit multimodal chain-of-reasoning: renders candidate regions as Set-of-Mark, MLLM generates/selects/refines prompts, SAM segments | Training-free; improves language-guided segmentation via interactive visual reasoning loop | Direct precedent for SoM-driven prompt refinement by an MLLM (2D natural images); closest architecture to ours |
| **Foundation Models for 3D Biomedical Image Segmentation** (2026 proceedings) | [10.1007/978-3-032-23496-4](https://doi.org/10.1007/978-3-032-23496-4) ⚠️ | MICCAI challenge proceedings (interactive 3D) | Metadata verified; individual chapters paywalled | Venue context; two chapters found via OpenAlex (ENSAM; Gaussian sampling) — not used for numbers |

**How corrective clicks are chosen in prior work.** (i) *Trained click policies*: SegAgent predicts the next click
with a PRM over mask state; PseudoClick predicts the next click directly (ECCV 2022, [2207.05282](#c-pseudoclick));
MedSAM-Agent generates expert-curated multi-turn trajectories filtered by ΔIoU and trains a process reward.
(ii) *Error-region sampling in training*: nnInteractive samples FP/FN components of the current prediction and
simulates point/box/scribble/lasso corrections (this is the mechanism our tool is already trained with).
(iii) *Clinician-driven rounds*: LeCor/SAMI3D-DW/SLIP evaluate corrective clicks under simulated or real users.
**Reported gains from iterative vs single prompt**: +0.22 category-macro Dice from 1→5 points and +0.04 more from
5 corrective clicks (SAMI3D-DW); SAM2 3× fewer interactions in video ([2408.00714](#c-sam2)); nnInteractive
interactive points >70 Dice average and lasso best; MedSAM2 >85% cost reduction in human-in-the-loop annotation.
**Gap for us**: none of these systems is an LLM agent driving nnInteractive with SoM pointing on 3D MRI; Seg-Agent
is 2D and training-free, VoxelPrompt drives a trained network by code, SlicerNNInteractive is a UI plugin.

---

## 4. Refinement tools that reduce the precision of pointing

| Tool / method | Link | How it works | Verified evidence | Could an agent use it? |
|---|---|---|---|---|
| **nnInteractive corrective prompts** | [2503.08373](https://arxiv.org/abs/2503.08373) ✅ PDF §2.3–2.4 | Follow-up prompts sampled from FP/FN components; previous interactions decayed ×0.9; network sees last prediction; simulated user agents (random/sunk-cost/single-style) | Trained on 120+ datasets; lasso AUC 83.42; interactive points >70 Dice; expert 3-slice scribbles 84.3 Dice | Already available; the agent just doesn't use it (E5–E11). Highest-value "tool" is the one we have |
| **3D Slicer "Grow from seeds"** | [Slicer Segment Editor docs](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html) ✅ page | Draw a seed inside each structure; improved **GrowCut** cell-automaton grows seeds to full segmentation; initialize/update loop | Docs: uses "an improved version of the grow-cut algorithm" (Zhu et al., MICCAI-IMIC 2014); at least two segments required | Yes: implement as "seed box/brush → local region-grow with intensity model"; gives full 3D extent from coarse seeds |
| **3D Slicer "Fill between slices"** | same Slicer page ✅ | Interpolate a complete segmentation between slices the user has segmented (any number of skipped slices) | Docs: "fill the skipped slices by interpolating between segmented slices" | Yes: agent segments first/last visible slice (or 2–3 slices) then interpolates — mirrors our z-end failure directly |
| **3D Slicer threshold painting / local threshold** | same Slicer page ✅ | Paint with an editable intensity range; "a large radius paint brush with threshold painting is often a very fast way to segment anatomy … partially connected to similar nearby structures" | Docs tip (verified) | Yes: expose "grow from a mark within intensity window" as a tool; cheap to implement server-side |
| **ITK-SNAP adaptive paintbrush + 4.4 AI** | [ITK-SNAP DLS quick start](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html) ✅; [ITK-SNAP main](https://www.itksnap.org/) ✅; classic tool paper [Yushkevich 2006](https://doi.org/10.1016/j.neuroimage.2006.01.015) ⚠️ | Adaptive paintbrush fills intensity-similar voxels under the brush; ITK-SNAP 4.4 sends paintbrush scribbles/polygons to nnInteractive for full AI segmentation | DLS docs (verified) describe nnInteractive integration; Yushkevich 2006 title reports "significantly improved efficiency and reliability" (metadata verified, abstract not fetched) | Yes: the adaptive-brush behaviour can be replicated by thresholded region growing; ITK-SNAP's own evolution shows the human workflow we are trying to give the agent |
| **Smart Brush** (2013) | [10.1109/ispa.2013.6703826](https://doi.org/10.1109/ispa.2013.6703826) ⚠️ | GPU sparse-field level-set solver; brush-driven real-time 3D surface evolution | Abstract (OpenAlex): "good, reliable segmentations in a short time and with few training"; validation study announced (wrist bones) | Conceptually yes (brush + level set); implementing level sets server-side is heavy |
| **Random walker** (2006, TPAMI) | [10.1109/TPAMI.2006.233](https://doi.org/10.1109/TPAMI.2006.233) ⚠️ | User labels a few pixels; each pixel assigned to label with max random-walk hitting probability; interactive multilabel | Abstract (OpenAlex): "small number of … labels … analytically and quickly"; high-quality segmentation | Algorithm for a seed-grow tool with soft boundaries; 3D random walker on full MRI would need downsampling/patch ROI |
| **Intelligent scissors / live-wire** (1995; 1997) | [10.1145/218380.218442](https://doi.org/10.1145/218380.218442) ⚠️; [10.1016/S1361-8415(97)85005-0](https://doi.org/10.1016/S1361-8415(97)85005-0) ⚠️ | Boundary "snaps" to strong edges along the optimal path between user clicks; live-wire for medical contours | Mortensen & Barrett (SIGGRAPH 1995), Barrett & Mortensen (MedIA 1997) — metadata verified; abstracts not fetchable (paywall) | A boundary-snapping tool reduces pointing precision for *delineation*; but our tumour boundaries are often weak/absent → lower priority than intensity region-grow |
| **PseudoClick** (2022, ECCV) | [2207.05282](https://arxiv.org/abs/2207.05282) ✅ abs | Model proposes the next (pseudo-)click on mislabelled regions; human only accepts/corrects | Alternative clicks reduce interaction cost vs. human-only clicks (abstract) | Direct template for an "auto-propose correction point" tool: propose k points along mask boundary/error-prone zones, let the agent pick |
| **RITM** (2022, preprint) | [2102.06583](https://arxiv.org/abs/2102.06583) ✅ abs | Feedforward click refinement with previous mask as input; can start from an external mask and *correct* it | Beat prior SOTA without inference-time optimisation; dataset choice matters | Confirms "correct an existing mask" is a first-class mode — our `initial-mask refinement` prompt path |
| **FocalClick** (2022, CVPR) | [10.1109/cvpr52688.2022.00136](https://doi.org/10.1109/cvpr52688.2022.00136) ⚠️ | Predicts/updates masks in localized crops; avoids destroying correct regions when refining | Abstract (OpenAlex): efficient on low-power devices; avoids destroying correct parts when refining preexisting masks | Concept of local refinement windows (matches our zoom-tool idea and crop-based correction) |
| **nnInteractive interaction/user simulation** (detailed) | [2503.08373](https://arxiv.org/abs/2503.08373) ✅ PDF §2.3–2.4 | FP/FN component sampling with Perlin fragmentation; slice sampling biased by foreground distribution; 3 simulated user styles | §2.3–2.4 (verified text) | Our auto-correction loop should mirror this: sample the largest error-prone region, propose a prompt there, decay old prompts |
| **Dynamic prompt generation for training** (2025) | [2510.03189](https://arxiv.org/abs/2510.03189) ✅ abs | Dynamic volumetric prompt generation + content-aware cropping during training; nnInteractive-initialised | Challenge Dice 0.6385 / NSD 0.6614 | Evidence that training-time interaction simulation is the standard; we consume such a model |
| **LeCor** (2026) | [2609.09477](https://arxiv.org/abs/2609.09477) ✅ abs | Correction rounds + test-time case adapters so a click improves *untouched* slices | 7 rounds: 0.787→0.827; 3 rounds ≈ 7 for the fine-tuned model; failures 47→27 | Sets realistic expectations for corrective rounds (~6-round plateau without adaptation); supports budgeting 2–3 rounds |
| **Human-efficiency anchors** | nnInteractive [2503.08373](https://arxiv.org/abs/2503.08373) ✅; MedSAM2 [2504.03600](https://arxiv.org/abs/2504.03600) ✅; MONAI Label [2203.12362](https://arxiv.org/abs/2203.12362) ✅ (also [MedIA 2024](https://doi.org/10.1016/j.media.2024.103207) ⚠️); DeepIGeoS [10.1109/TPAMI.2018.2840695](https://doi.org/10.1109/TPAMI.2018.2840695) ⚠️; BIFSeg [10.1109/TMI.2018.2791721](https://doi.org/10.1109/TMI.2018.2791721) ⚠️ | Interactive AI tools vs manual annotation | nnInteractive −72% time (179 vs 635 s); MedSAM2 >85% cost reduction; MONAI Label framework for AI-assisted interactive labelling; DeepIGeoS/BIFSeg reduce interactions when correcting CNN output | These are the human baselines the agent should approach per case; the agent's *own* workflow should imitate the fastest human loop (few, well-placed corrections) |

**Which of these to expose to the agent (ranked).**
1. *nnInteractive corrective prompts* (zero implementation; already there; the agent under-uses it).
2. *Extent seeding* (first/last slice scribbles → nnInteractive 3D propagation; ≈ "Fill between slices" but native).
3. *Zoom/crop + re-render* (see §5; reduces the pointing-precision problem rather than adding algorithms).
4. *Local threshold + region grow from a mark* (Slicer/ITK-SNAP adaptive brush analogue; cheap: intensity model
   from a sphere around the mark, threshold, connected component; ROI-limited 3D).
5. *Multi-slice interpolation* (segment 2–3 slices, interpolate, then refine with nnInteractive; mirrors Fill
   between slices).
6. *Boundary snapping / live-wire* (only if delineation, not detection, becomes the bottleneck; weak boundaries
   in soft-tissue MRI limit the payoff).

---

## 5. Small/subtle lesion localization by VLMs

| Work (year, venue) | Link | Mechanism | Key verified numbers | Relevance to us |
|---|---|---|---|---|
| **V\*** (2023) | [2312.14135](https://arxiv.org/abs/2312.14135) ✅ abs | LLM-guided visual search: LLM proposes regions of interest, crops them, re-queries; SEAL meta-architecture; V*Bench | Introduces visual search to MLLMs; improves high-resolution/crowded-image performance (abstract) | The core idea behind a zoom tool: use the model's world knowledge to pick a region, then re-render |
| **ZoomEye** (2024→2025, EMNLP 2025 Main) | [2411.16044](https://arxiv.org/abs/2411.16044) ✅ abs | Training-free, model-agnostic tree search over zoomed sub-regions (image as hierarchy) | Improves a series of high-resolution benchmarks (abstract; accepted EMNLP 2025 Main) | Cheap (no training) zoom policy the agent can call per uncertain case |
| **DeepEyes** (2025) | [2505.14362](https://arxiv.org/abs/2505.14362) ✅ abs | RL without cold-start SFT teaches a VLM to "think with images" (active perception / cropping) | Emergent grounding-based zooming via RL | Long-term evidence that image zooming is learnable behaviour; we can only prompt it |
| **Zoom-Refine** (2025) | [2506.01663](https://arxiv.org/abs/2506.01663) ✅ abs | Training-free localized zoom + self-refinement protocol | Improves high-resolution MLLM understanding (abstract) | Same tool pattern; relevant for cropped re-render design |
| **Chain-of-Focus (CoF)** (2025) | [2505.15436](https://arxiv.org/abs/2505.15436) ✅ abs | Adaptive focusing/zooming driven by visual cues; SFT on MM-CoF 3K + RL; outcome+format rewards | +5% over existing VLMs on the V* benchmark across 8 resolutions 224→4K; repo verified | Quantifies the resolution-dependent gain we hope to get on F15/L33-type cases |
| **CoFFT** (2025) | [2509.22010](https://arxiv.org/abs/2509.22010) ✅ abs | Training-free "foresight-focus" thought: sample diverse reasoning paths, pick by visual focus, adjust focus, iterate | Training-free alternative to CoF | A test-time-only recipe compatible with our no-training constraint |
| **Set-of-Mark prompting** (2023) | [2310.11441](https://arxiv.org/abs/2310.11441) ✅ abs | Overlay numbered/alphanumeric marks on segmented regions; GPT-4V can then ground and reference regions | SoM zero-shot GPT-4V outperforms a fully-finetuned referring-expression model on RefCOCOg (abstract) | We already use SoM; the paper establishes the mechanism. **Mark-density effects: ❓ no verified study found** |
| **Dragonfly** (2024) | [2406.00977](https://arxiv.org/abs/2406.00977) ✅ abs | Native-resolution + multi-crop "zoom-in encoding" beyond native resolution; mean-pooled tokens; Dragonfly-Med | Top ranks across ten 7–8B benchmarks; strong TextVQA/ChartQA; biomedical variant | Evidence for tiling/multi-crop as an alternative to single-image downsampling |
| **DeepTumorVQA** (2025, NeurIPS 2025 D&B) | [2505.18915](https://arxiv.org/abs/2505.18915) ✅ abs | 3D tumour VQA benchmark over CT volumes; recognition/measurement/reasoning | 9,262 CT volumes / 3.7M slices / 395K questions; four VLMs (RadFM, M3D, Merlin, CT-CHAT) struggle with **small tumour detection** and lesion recognition; preprocessing/vision design matters | The best verified statement that small-lesion detection is a medical-VLM weak point; supports both zoom and 3D-context investments |
| **MedSG-Bench** (2025, NeurIPS 2025) | [10.52202/085713-2502](https://doi.org/10.52202/085713-2502) ⚠️ | First medical visual grounding benchmark on image *sequences* | Metadata verified; abstract not fetched | Metric/benchmark context for medical grounding (not used for numbers) |
| **Look-Closer-Then-Diagnose** (2026) | [2605.21652](https://arxiv.org/abs/2605.21652) ✅ abs | Structured Zoom-then-Diagnose paradigm for ultrasound VQA; GRPO with uncertainty-aware reward from stochastic rollouts | "Improves lesion localization by 39.3%" (reported); liver/breast/thyroid datasets | Medical VLM zooming is being adopted; the 39.3% is their reported relative localization improvement, on ultrasound VQA — not directly transferable to our Dice setting |
| **Effect of image resolution** (2020, Radiology AI) | [10.1148/ryai.2019190015](https://doi.org/10.1148/ryai.2019190015) ⚠️ | Resolution sweep 32×32 → 600×600 for CNN chest-radiograph diagnosis | Abstract (OpenAlex): 112,120 images / 30,805 patients; ResNet34/DenseNet121; performance varies with resolution | Classic evidence that downsampling costs diagnostic performance; motivates native-resolution crops for subtle lesions |
| **Volumetric radiology MLLM review** (2026) | [2608.20549](https://arxiv.org/abs/2608.20549) ✅ abs | Review of 200+ works on volumetric representation vs. 2D-view MLLM reasoning; agentic orchestration | Distinguishes when selected 2D views suffice vs. when native volumetric modelling is needed | Frames our central tension: the agent sees rendered 2D views; the task needs 3D context (extent, z-ends) |
| **Seg-Agent SoM loop** (2026) | [2605.12953](https://arxiv.org/abs/2605.12953) ✅ abs | SoM rendering of candidate regions inside an interactive generate/select/refine loop | Training-free improvements on language-guided segmentation | Shows SoM is not only for pointing but for *search/selection* in an agent loop |
| **Medical VLM surveys** (2026) | [10.3390/bioengineering13070803](https://doi.org/10.3390/bioengineering13070803) ⚠️; [2506.18378](https://arxiv.org/abs/2506.18378) ⚠️(not fetched) | Reviews of language-guided medical segmentation / VLM medical analysis | Metadata only | Context, not evidence |

**What the evidence supports for our setting.** (a) Downsampling and single-view rendering are established causes of
missed subtle findings, and crop/zoom at test time is the most portable fix; (b) all test-time zoom methods that
work under our constraints (no fine-tuning) are *training-free tree/loop search* (ZoomEye, CoFFT, Zoom-Refine) or
RL-trained cropping we cannot reproduce (DeepEyes, CoF); (c) the VLM pointing ceiling we observe (~0.90 on a
correct slice) is consistent with published medical grounding limitations — the leverage is in *which* pixels the
render shows and at what effective resolution, not in more histotype knowledge (E10); (d) **no study was found on
how set-of-mark count/density affects VLM pointing accuracy — this is an open question we can answer cheaply
ourselves.**

---

## 6. Proposed experiments for our harness (design, cost, risk)

All harness changes must be recorded in `research/experiment_log.md`; the 30 design cases are already used, so
treat them as a development set with an internal 15/15 (histotype-stratified) split and report final results only
on the 169-case test split after freezing.

**E-A. Forced corrective-prompt loop (priority 1).**
*Hypothesis*: the un-used refinement channel is worth more than any localisation tweak; ≥1 corrective prompt per
case raises mean Dice on under-segmented cases (M33/N34/N01-type) without hurting localised ones.
*Design*: (i) hard rule in the system prompt: after the first mask and the review, issue at least one
`seg_point`/`seg_scribble` targeted at (a) a cranio-caudal end and (b) a locally under-included region, then
re-review; `seg_finish` only after a second review. (ii) Add `seg_autocorrect` that computes FP/FN-like candidate
regions *without GT* — e.g., low-intensity boundary zones and the two slices immediately past each mask end — and
returns 3 suggested marks; the agent picks one. Optionally mirror nnInteractive's simulation (error components,
decay ×0.9). *Read-out*: prompts/case, mean Dice, recall/precision, fraction of cases where the second mask changes
Dice >0.02. *Cost*: harness + prompt work; +1–2 tool calls and +1 model turn per case (~+20–50% wall-clock;
API-equivalent ≈ $0.05–0.25/case at xhigh). *Risk*: forced prompts can hurt (F32 over-segmentation) → require a
justification path and keep a no-op option; use paired comparisons per case.

**E-B. Zoom/crop observation tool (priority 2).**
*Hypothesis*: the 2 mL lesion (F15), the fat-suppressed lipoma (L33) and the subtle lesion beside a dark structure
(F14) are resolvable at higher effective resolution; the Phase 2 plateau is a rendering limitation, not knowledge.
*Design*: `seg_zoom(mark_id | box, scale 2–4×)` renders a cropped window of the current slice (or an orthogonal
plane) with marks recomputed inside the crop; the agent is prompted to zoom when the target is small/subtle or
partially outside the mask. Evaluate on 15/15 split with hard-case enrichment. *Cost*: render path exists; new
crop+mark path moderate; +1–3 views per case (~$0.05–0.3/case equivalent). *Risk*: mark geometry in crops must be
validated (existing coverage tests can be reused); zoom can waste budget if used eagerly → enable only under
conditions (target < X mL or localisation failure signal).

**E-C. Two-plane extent seeding (priority 3).**
*Hypothesis*: our systematic z-end under-segmentation is fixed by prompting on first+last tumour slice rather than
a single mid-slice point. *Design*: prompt rule + tool `seg_scribble_planes(along z)`; compare (i) current, (ii)
forced 2-plane scribble, (iii) 3-slice interpolation analogue (segment 2 slices, nnInteractive-propagate).
*Cost*: very low (existing prompt types). *Risk*: nnInteractive prompt semantics for out-of-plane scribbles must
be checked (`harness/server/CONTRACTS.md`); scribble quality may vary.

**E-D. Local region-grow / adaptive-brush tool (priority 4).**
*Hypothesis*: for hypointense/iso-intense tumour parts, a threshold+connected-component grow inside a mark-derived
box includes more true tumour than point prompts. *Design*: `seg_grow(mark, intensity_window="auto|given")` in the
segd daemon: estimate intensity distribution in a sphere around the mark, threshold, 3D connected component within
a dilated box; return mask + volume; agent may then refine with nnInteractive. Inspired by Slicer Grow-from-seeds /
ITK-SNAP adaptive brush. *Cost*: medium (server-side implementation + tests). *Risk*: leaks into adjacent
similar-intensity structures (F32-style over-segmentation); mitigate with box constraints and post-hoc nnInteractive
negative points.

**E-E. External verifier before finish (priority 5).**
*Hypothesis*: a checker that is not the same reasoning turn (deterministic QC and/or VLM judge on renders) reduces
"confirm without editing". *Design*: (i) deterministic QC block: component count, volume, z-extent vs. body, mask
boundary intensity contrast, "signal present beyond both ends?"; (ii) optionally a single VLM call on the 3×5
review grid with the checklist (independent of the agent's own turn), returning `ok / needs correction + reason`;
(iii) hard rule: cannot finish while QC flags a possible missed extent. *Cost*: 1 extra cheap call/case (~$0.02–0.1
equivalent). *Risk*: verifier hallucination ([CriticGPT 2407.00215](#c-criticgpt)); prompt sensitivity
([MOTIVE](#c-motive)); use multiple criteria/views and require concrete evidence (mark ids) for a veto.

**E-F. GEPA on the system prompt with a stratified split (priority 6, after E-A–E-E).**
*Hypothesis*: reflective prompt evolution finds a better instruction set than our hand-written passes (which did
not change behaviour, E9). *Design*: split 30 design cases into 15 train / 15 validation (3 histotypes × 5 each,
or 2-fold with 3 folds); objective = mean Dice (maybe + localisation-failure penalty); feedback = per-case Dice +
the agent transcript + (optionally) the render/CMVF-style visual diagnosis; budget 50–150 candidate evaluations on
minibatches of 5 (250–750 case-runs); candidate selection on validation; then freeze and evaluate on the 169-case
test split. *Cost*: 250–750 xhigh case-runs ≈ $60–190 API-equivalent plus reflection calls (quota-heavy under
subscription: schedule over days). *Risk*: overfitting to 30 cases (mitigate: held-out validation, Pareto, freeze
before test); validation rollouts dominate budget ([GEPA](#c-gepa)); run-to-run noise (E9) → require paired
improvements, and report CIs ([2411.00640](#c-errorbars)). Alternative/complement: ACE-style playbook memory
([2510.04618](#c-ace)) — inspectable, no-label, cheaper, but less search power.

**E-G. SoM density / rendering ablation (priority 7, cheap).**
*Hypothesis*: the ~0.90 single-slice ceiling and the F14/F15/L33 misses are partly a mark-density/visibility
artifact. *Design*: re-run the Phase 2 S condition with 2–3 mark densities (coarse/current/dense) and crop-aware
marks, on the 30 design cases; measure hit rate and near-miss distance. *Cost*: 3×90 cheap calls ≈ small.
*Risk*: none structural; the result may simply confirm a genuine model ceiling (which is itself valuable and
currently the main threat to the whole approach).

**Measurement standard for all of the above.** Pre-register the gate, run paired per-case comparisons, use
bootstrap CIs over cases (30-case means have ±0.03–0.05 Dice run-to-run; E9), and keep every change in the
experiment log. The Phase 2 gate (S ≥ 0.80 and S−R ≥ 0.10) is the model to follow.

---

## 7. Citation list (all links opened/verified in this session unless marked)

**Prompt/program optimisation.**
<a id="c-gepa"></a>1. ✅ GEPA — Agrawal et al., *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, ICLR 2026 (Oral); arXiv:2507.19457 — https://arxiv.org/abs/2507.19457 (abs + PDF fetched).
2. ✅ DSPy — Khattab et al., *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*, preprint (PDF header says "Preprint"); arXiv:2310.03714 — https://arxiv.org/abs/2310.03714 (abs + PDF).
3. ✅ MIPROv2 — Opsahl-Ong et al., *Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs*, EMNLP 2024; arXiv:2406.11695 — https://arxiv.org/abs/2406.11695 (abs + PDF).
4. ✅ SIMBA — DSPy optimizer documentation (no standalone paper found) — https://dspy.ai/api/optimizers/SIMBA/; behaviour comparison in arXiv:2602.17937 — https://arxiv.org/abs/2602.17937.
5. ✅ TextGrad — Yuksekgonul et al., *TextGrad: Automatic "Differentiation" via Text*, preprint; arXiv:2406.07496 — https://arxiv.org/abs/2406.07496 (abs + PDF). Note: the paper later appeared in Nature (2025) but that venue was not verified in this review.
6. ✅ OPRO — Yang et al., *Large Language Models as Optimizers*, ICLR 2024; arXiv:2309.03409 — https://arxiv.org/abs/2309.03409.
7. ✅ Promptbreeder — Fernando et al., *Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution*, arXiv:2309.16797 — https://arxiv.org/abs/2309.16797 (PDF).
<a id="c-ace"></a>8. ✅ ACE — Zhang et al., *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models*, ICLR 2026; arXiv:2510.04618 — https://arxiv.org/abs/2510.04618 (abs + PDF).
9. ✅ Trace / OptoPrime — Cheng et al., *Trace is the Next AutoDiff: Generative Optimization with Rich Feedback, Execution Traces, and LLMs*, arXiv:2406.16218 — https://arxiv.org/abs/2406.16218 (abs + PDF).
10. ✅ AFlow — Zhang et al., *AFlow: Automating Agentic Workflow Generation*, ICLR 2025 (Oral); arXiv:2410.10762 — https://arxiv.org/abs/2410.10762 (PDF).
11. ✅ ADAS — Hu et al., *Automated Design of Agentic Systems*, arXiv:2408.08435 — https://arxiv.org/abs/2408.08435 (PDF).
12. ✅ Adaptive-GEPA — arXiv:2609.38762 — https://arxiv.org/abs/2609.38762 (abs).
13. ✅ Ranking-aware prompt evolution for clinical MLLMs (Ranking-PE) — arXiv:2609.40361 — https://arxiv.org/abs/2609.40361 (abs).
14. ✅ Cross-Modal Visual Feedback for APO (CMVF) — arXiv:2607.24354 — https://arxiv.org/abs/2607.24354 (abs).
15. ✅ GEPA applied to RCT risk-of-bias assessment — arXiv:2512.01452 — https://arxiv.org/abs/2512.01452 (abs).
16. ✅ Prompt Triage (DSPy for medical VLMs) — arXiv:2511.11898 — https://arxiv.org/abs/2511.11898 (abs).
17. ✅ Benchmarking APO with chess (frozen LLMs) — arXiv:2610.00416 — https://arxiv.org/abs/2610.00416 (abs).
18. ✅ Analyzing LLM instruction optimization for tabular fact verification (COPRO/MIPROv2/SIMBA) — arXiv:2602.17937 — https://arxiv.org/abs/2602.17937 (abs).

**Agents, self-correction, verification.**
19. ✅ Reflexion — Shinn et al., NeurIPS 2023; arXiv:2303.11366 — https://arxiv.org/abs/2303.11366.
20. ✅ Self-Refine — Madaan et al., NeurIPS 2023; arXiv:2303.17651 — https://arxiv.org/abs/2303.17651.
21. ✅ CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing — ICLR 2024 (arXiv listing comment); arXiv:2305.11738 — https://arxiv.org/abs/2305.11738.
22. ✅ *Large Language Models Cannot Self-Correct Reasoning Yet* — Huang et al., ICLR 2024; arXiv:2310.01798 — https://arxiv.org/abs/2310.01798.
23. ✅ Chain-of-Verification (CoVe) — Dhuliawala et al., arXiv:2309.11495 — https://arxiv.org/abs/2309.11495.
<a id="c-s1"></a>24. ✅ s1: Simple test-time scaling (budget forcing) — Muennighoff et al., arXiv:2501.19393 — https://arxiv.org/abs/2501.19393.
<a id="c-underthink"></a>25. ✅ *Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs* — Wang et al., arXiv:2501.18585 — https://arxiv.org/abs/2501.18585.
26. ✅ OptimalThinkingBench — arXiv:2508.13141 — https://arxiv.org/abs/2508.13141.
<a id="c-ttsagents"></a>27. ✅ Evaluating Test-Time Scaling of General LLM Agents — arXiv:2602.18998 — https://arxiv.org/abs/2602.18998.
28. ✅ *Let's Verify Step by Step* (PRM800K dataset) — arXiv:2305.20050 — https://arxiv.org/abs/2305.20050 (venue not verified in this review).
<a id="c-aaj"></a>29. ✅ Agent-as-a-Judge — Zhuge et al., arXiv:2410.10934 — https://arxiv.org/abs/2410.10934.
<a id="c-criticgpt"></a>30. ✅ CriticGPT — arXiv:2407.00215 — https://arxiv.org/abs/2407.00215.
<a id="c-segae"></a>31. ✅ SegAE (VLM judge for medical segmentation labels) — arXiv:2601.14406 — https://arxiv.org/abs/2601.14406.
<a id="c-motive"></a>32. ✅ MOTIVE (multi-view VLM self-verification) — arXiv:2610.07018 — https://arxiv.org/abs/2610.07018.
33. ✅ Model-Adaptive Tool Necessity / knowing–doing gap — arXiv:2605.14038 — https://arxiv.org/abs/2605.14038.
34. ✅ Tunable Tool-Call Rates in LLM Agents via Representation Steering — arXiv:2608.25198 — https://arxiv.org/abs/2608.25198.
35. ✅ Why Do Multi-Agent LLM Systems Fail? (MAST) — arXiv:2503.13657 — https://arxiv.org/abs/2503.13657.
<a id="c-errorbars"></a>36. ✅ Adding Error Bars to Evals — arXiv:2411.00640 — https://arxiv.org/abs/2411.00640.

**Agentic / LLM-driven interactive segmentation.**
37. ✅ SegAgent — arXiv:2503.08625 (CVPR 2025) — https://arxiv.org/abs/2503.08625.
38. ✅ Seg-Zero — arXiv:2503.06520 — https://arxiv.org/abs/2503.06520.
39. ✅ Seg-R1 — arXiv:2506.22624 — https://arxiv.org/abs/2506.22624.
40. ✅ SegLLM — arXiv:2410.18923 — https://arxiv.org/abs/2410.18923.
41. ✅ LISA — arXiv:2308.00692 (CVPR 2024) — https://arxiv.org/abs/2308.00692.
42. ✅ VoxelPrompt — arXiv:2410.08397 — https://arxiv.org/abs/2410.08397.
43. ✅ MedRAX — arXiv:2502.02673 — https://arxiv.org/abs/2502.02673.
44. ✅ MedSAM-3 / MedSAM-3 Agent — arXiv:2511.19046 — https://arxiv.org/abs/2511.19046.
45. ✅ MedSAM-Agent — arXiv:2602.03320 — https://arxiv.org/abs/2602.03320.
46. ⚠️ MediRound (multi-round medical reasoning segmentation) — LNCS, DOI 10.1007/978-3-032-37196-6_21 (OpenAlex metadata verified; text paywalled).
<a id="c-medsam2"></a>47. ✅ MedSAM2 — arXiv:2504.03600 — https://arxiv.org/abs/2504.03600.
48. ✅ Segment Anything (SAM) — arXiv:2304.02643 — https://arxiv.org/abs/2304.02643.
<a id="c-sam2"></a>49. ✅ SAM 2 — arXiv:2408.00714 — https://arxiv.org/abs/2408.00714.
<a id="c-nnint"></a>50. ✅ nnInteractive — arXiv:2503.08373 — https://arxiv.org/abs/2503.08373 (abs + full PDF).
51. ✅ SlicerNNInteractive — arXiv:2504.07991 — https://arxiv.org/abs/2504.07991.
<a id="c-sami3d"></a>52. ✅ SAMI3D-DW — arXiv:2609.25743 — https://arxiv.org/abs/2609.25743.
53. ✅ SLIP — arXiv:2607.22332 — https://arxiv.org/abs/2607.22332.
<a id="c-lecor"></a>54. ✅ LeCor — arXiv:2609.09477 — https://arxiv.org/abs/2609.09477.
55. ✅ Dynamic Prompt Generation for Interactive 3D Medical Image Segmentation Training — arXiv:2510.03189 — https://arxiv.org/abs/2510.03189.
<a id="c-segagent2"></a>56. ✅ Seg-Agent (training-free, SoM interactive loop) — arXiv:2605.12953 — https://arxiv.org/abs/2605.12953.
57. ⚠️ Foundation Models for 3D Biomedical Image Segmentation (MICCAI challenge proceedings, LNCS 2026) — DOI 10.1007/978-3-032-23496-4 (OpenAlex metadata verified).

**Refinement tools / human efficiency.**
<a id="c-slicerdocs"></a>58. ✅ 3D Slicer Segment Editor documentation (Grow from seeds, Fill between slices, Threshold painting) — https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html (page fetched).
<a id="c-itksnapdls"></a>59. ✅ ITK-SNAP Deep Learning Segmentation quick start (nnInteractive integration, v4.4.0) — https://itksnap-dls.readthedocs.io/en/latest/quick_start.html (page fetched).
58a. ⚠️ Fast GrowCut — Zhu, Kolesov, Gao, Kikinis, Tannenbaum (MICCAI-IMIC workshop 2014). Citation taken from the Slicer Segment Editor docs (secondary source); the original workshop paper was not fetched in this review. <a id="c-growcut"></a>
60. ⚠️ ITK-SNAP — Yushkevich et al., *User-guided 3D active contour segmentation of anatomical structures…*, NeuroImage 2006, DOI 10.1016/j.neuroimage.2006.01.015 (OpenAlex metadata; abstract not fetchable).
61. ⚠️ Smart Brush — DOI 10.1109/ispa.2013.6703826 (OpenAlex abstract).
62. ⚠️ Random Walks for Image Segmentation — Grady, IEEE TPAMI 2006, DOI 10.1109/TPAMI.2006.233 (OpenAlex abstract).
63. ⚠️ Intelligent Scissors for Image Composition — Mortensen & Barrett, SIGGRAPH 1995, DOI 10.1145/218380.218442 (OpenAlex metadata).
64. ⚠️ Interactive live-wire boundary extraction — Barrett & Mortensen, Medical Image Analysis 1997, DOI 10.1016/S1361-8415(97)85005-0 (OpenAlex metadata + author list verified; full text paywalled).
<a id="c-pseudoclick"></a>65. ✅ PseudoClick — arXiv:2207.05282 (ECCV 2022) — https://arxiv.org/abs/2207.05282.
66. ✅ RITM — arXiv:2102.06583 — https://arxiv.org/abs/2102.06583.
67. ⚠️ FocalClick — CVPR 2022, DOI 10.1109/cvpr52688.2022.00136 (OpenAlex abstract).
68. ⚠️ DeepIGeoS — Wang et al., IEEE TPAMI 2018, DOI 10.1109/TPAMI.2018.2840695 (OpenAlex abstract).
69. ⚠️ BIFSeg — Wang et al., IEEE TMI 2018, DOI 10.1109/TMI.2018.2791721 (OpenAlex abstract).
70. ✅ MONAI Label — arXiv:2203.12362 — https://arxiv.org/abs/2203.12362; journal version: Medical Image Analysis 2024, DOI 10.1016/j.media.2024.103207 (OpenAlex metadata; publisher page paywalled).
71. ⚠️ 3D Slicer platform — Fedorov et al., Magnetic Resonance Imaging 2012, DOI 10.1016/j.mri.2012.05.001 (OpenAlex metadata).

**VLM localisation / zoom / resolution.**
<a id="c-vstar"></a>72. ✅ V\* — arXiv:2312.14135 — https://arxiv.org/abs/2312.14135.
<a id="c-zoomeye"></a>73. ✅ ZoomEye — arXiv:2411.16044 (EMNLP 2025 Main) — https://arxiv.org/abs/2411.16044.
<a id="c-deepeyes"></a>74. ✅ DeepEyes — arXiv:2505.14362 — https://arxiv.org/abs/2505.14362.
75. ✅ Zoom-Refine — arXiv:2506.01663 — https://arxiv.org/abs/2506.01663.
<a id="c-cof"></a>76. ✅ Chain-of-Focus — arXiv:2505.15436 — https://arxiv.org/abs/2505.15436.
77. ✅ CoFFT — arXiv:2509.22010 — https://arxiv.org/abs/2509.22010.
78. ✅ Set-of-Mark Prompting — arXiv:2310.11441 — https://arxiv.org/abs/2310.11441.
79. ✅ Dragonfly — arXiv:2406.00977 — https://arxiv.org/abs/2406.00977.
<a id="c-deeptumorvqa"></a>80. ✅ DeepTumorVQA — arXiv:2505.18915 — https://arxiv.org/abs/2505.18915.
81. ⚠️ MedSG-Bench — NeurIPS 2025, DOI 10.52202/085713-2502 (OpenAlex metadata).
82. ✅ Look-Closer-Then-Diagnose (active zooming, ultrasound VQA) — arXiv:2605.21652 — https://arxiv.org/abs/2605.21652.
83. ⚠️ The Effect of Image Resolution on Deep Learning in Radiography — Radiology AI 2020, DOI 10.1148/ryai.2019190015 (OpenAlex abstract).
84. ✅ Volumetric Radiology AI in the Era of Multimodal LLMs (review) — arXiv:2608.20549 — https://arxiv.org/abs/2608.20549.
85. ⚠️ Taming Vision-Language Models for Medical Image Analysis (review) — arXiv:2506.18378 (listed but not fetched; marked unverified).
86. ⚠️ Language-Guided Segmentation of Medical Images: A Review of Foundation Models — Bioengineering 2026, DOI 10.3390/bioengineering13070803 (OpenAlex metadata).

**Gaps found (❓, explicitly unanswered by this search).**
- No published study of **set-of-mark density/count effects** on VLM pointing accuracy was found; our Phase 2 data
  (S ≈ 0.89–0.91 stable across conditions) is currently the best local evidence.
- No published **GEPA/DSPy/TextGrad/OPRO/ACE application to an interactive segmentation agent** was found; the
  closest are medical VLM prompt optimisation (Prompt Triage), clinical ranking (Ranking-PE), VQA (CMVF), and
  segmentation agents that rely on training (SegAgent, Seg-Zero, MedSAM-Agent).
- No controlled comparison of **LLM-agent-driven refinement vs. plain nnInteractive refinement** on 3D MRI was
  found; SAMI3D-DW and SLIP provide the closest simulated/real-user interaction baselines.
