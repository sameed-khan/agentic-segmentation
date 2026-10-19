# Decision models / "System 1" models: Jev and the fast–slow decision paradigm

**Scope.** Focused web research for the agentic 3D-segmentation project. `optimization_and_refinement_review.md` covers prompt
optimization and interactive-segmentation refinement; this note covers the *decision-model* paradigm. Questions: (1) what "Jev" is;
(2) the surrounding fast/slow-agent paradigm and how it is evaluated; (3) whether a decision model fits our harness, where **every action is
a choice among enumerated options** (numbered marks on rendered slices, enumerated `seg_*` tools, and the planned C1..Cn candidate
corrections the agent accepts/rejects).

**How this was produced (verification policy).** Date of search **2026-10-19**. The `pi-web-access` MCP tools were *not* available in this
session; sources were fetched directly with `curl` (vendor sites, the `.md` variants of the Mintlify docs, arXiv abs/HTML pages) plus the
public arXiv API and package registries. **✅ = primary source fetched in this session** (quoted numbers come from it). **⚠️ = claim known
only from a secondary mention, not fetched.** **❓ = nothing verified found** (explicit gap). Statements about *our* harness are labelled
**[speculation]** — this note makes no claim that has not been fetched or explicitly flagged.

---

## 1. What "Jev" is

**Vendor and release.** Jev is the first public **System One model** from **TypeSafe AI** (homepage: "Made in SF"), announced
**2026-09-15** in early access. Current versions at search date: `jev-1.13.0` and `jev-1.14.0` (released 2026-10-14); aliases
`jev-latest` and `jev-preview`. ✅ [TS-blog-intro] [TS-blog-1.14] [TS-docs-models]. Author of the announcement: "Diogo Almeida,
founder"; the blog says he worked at OpenAI on instruction-following/RLHF research behind ChatGPT (self-description, not independently
verified ⚠️).

**What it is.** A **non-generative** model: it never emits text. Given a `state` and typed `questions`, it returns a choice from a fixed
option set, a position on a rubric, or the probability that a statement is true, each with a probability distribution and (for some types)
a `confidence`. ✅ [TS-blog-intro] [TS-docs-intro] [arXiv-2609.37647].

**Architecture.** Vendor claims: "a new architecture, a new sampler [parallel sampler], and a new training algorithm we call
Reinforcement Learning for Calibrated Decisions (RLCD)"; outputs are produced "in a single query" instead of token-by-token. Parameter
count, layer structure, tokenizer and training data are **not disclosed** in anything I could fetch; the homepage FAQ renders its answers
only in the browser ("Is Jev just a smaller LLM?", "Where does our training data come from?") — ❓. ✅ [TS-blog-intro] [TS-home].

**API, inputs, outputs.** `POST https://api.typesafe.ai/v1/systemone` (unauthenticated call returns 403, i.e. the endpoint is live);
`state` is a string, JSON object, or array of text; `questions` is a map of named typed questions. Primitives ✅ [TS-docs-models] [TS-docs-intro]:
`choice` (options map, **up to 255 options**), `score` (ordered levels), `noul` (probability 0–1), `box` (bounding box as fractions,
**`jev-1.14.0` only**, requires an image). Answers carry the typed value, a probability per option, and `confidence`. No text output;
output tokens are free. Official clients: PyPI `typesafe-sdk` 0.7.5, npm `@typesafe-ai/sdk` 0.7.0 (registry metadata fetched ✅).

**Latency.** Vendor: end-to-end **70–500 ms**; "40x–200x faster" than frontier LLMs on System-One-shaped queries, homepage claim
"193.6x Faster, 444.6x Cheaper … based on workflows for System One tasks" ($0.000081 / 0.114 s vs $0.013880 / 8.566 s). Independent:
**321 ms per request** through the gateway (closed LLMs 1.3–6.7 s; local System-1 models 41–228 ms). ✅ [TS-blog-intro] [TS-home]
[arXiv-2610.21244].

**Pricing.** Vendor: **$42 per billion input tokens = $0.042/Mtok**, output free; homepage comparison "238x lower input price than
Claude Fable 5.1". Independent measurement: **$0.052 per 1,000 requests** (GPT-6 Luna $0.14/1k; Microsoft-Decision-1 $0.038/1k).
Context: 64k per request / 32k per question, raised to 262k with a "danger zone" header. ✅ [TS-docs-models] [TS-home] [arXiv-2610.21244].

**Multimodality (critical for us).** Images may be placed anywhere text may go (`state`, a question's `instructions`, its `criteria`)
behind the header `TypeSafe-Danger-Zone-Images: true`; the vendor says `jev-1.13.0` "was multimodal all along" and that both 1.13/1.14
accept images. It is explicitly **not production-grade**: "Images and Box questions aren't yet up to our standards to call ready for
production", "accuracy will change as we update the models", and "all models (Jev included) perform worse on multimodality over pure
text … breaking down questions into the simplest possible decompositions helps a lot". Image limits: PNG/JPEG only, 280–1568 px per side,
≤16 MiB, 32 MiB per expanded request, image tokens billed as input tokens (estimation guidance "soon" ❓). ✅ [TS-docs-images] [TS-blog-1.14].
**Documentation contradiction (same day, both fetched):** `docs.typesafe.ai/concepts/system-one` and `.../models` state "Jev currently
accepts text input only. Images, audio, and video are not supported (yet)", while `.../images`, `.../primitives/box` and the 1.14 blog
document image input. Treat image support as real but experimental and version-drifting.

**Benchmarks / evaluations.**
- Vendor: four "workflow evals" scored against the average of GPT-6 Astra and Fable 5.1 (no published paper by the vendor) ✅ [TS-blog-intro].
- Independent, **arXiv:2609.37647** (Deußer, Sparrenberg, Sifa; 2026-09-29): Jev 1.13.0 zero-shot on **37 datasets, 346,009 requests,
  under USD 10 total**; 95–99% on IMDB/SST-2/HellaSwag/ARC; 86.7% on Belebele (122 languages); beats Qwen3.8-27B on 27/37 datasets and
  Gemma-4-E4B on all 37; choice probabilities well calibrated; binary probabilities poorly placed relative to a fixed 0.5 threshold;
  degrades on low-resource languages, fine-grained/noisy labels, rubric judgments; rotating the options leaves accuracy unchanged. ✅
- Independent, **arXiv:2610.21244** (System1-Arena, UT Austin; 2026-10-15): Jev **72.6%** accuracy on non-agentic decisions (GPT-6 Luna
  71.2%), Brier **0.349**; **67.8%** single-step agentic accuracy but only **17.4%** closed-loop success. ✅
- Third-party benchmarks JevBench (Benchmark Heaven) and ReflexBench (Brida AI) exist per System1-Arena's related work, but I could not
  fetch them ⚠️. Jev's own versioned docs include a vendor-written failure-mode page ✅ [TS-docs-jagged].
- **Model card:** no separate model card or vendor paper found ❓; the docs pages, the "Jev 1.13 jaggedness" page, `GET /v1/models`
  metadata and the two release blog posts are the closest thing.

---

## 2. The paradigm: "decision models" / System-1 models, 2025–2026

**Who ships the category.** TypeSafe's *System One* (Jev) ✅; **OpenAI Decisions API** — released in beta **2026-10-06**, `gpt-6-luna`
only, dedicated `POST /v1/decisions`, typed `predicate` / `choice` / `score` questions, **accepts text and images**, "10x faster than the
Responses API", $0.10/Mtok short-context input (standard tier), documented alternative to Responses/structured outputs ✅ [OAI-decisions]
[OAI-changelog] [OAI-pricing]; **Microsoft-Decision-1** — Foundry deployment ("classification, routing, ranking, grading, and binary
decisions"), `noul`/`choice`/`score` over a text-or-JSON state, described as a "system-one API" shape, with no image input claimed ✅
[MS-learn]; System1-Arena reports that model as post-trained from Qwen3.5-9B and lists the rest of the open-weight decision models —
Laya (Convai Innovations), CLM (Kwok et al.), five "Decision 2.0" models (vLLM Semantic Router team) — giving 10 System-1 models vs
20 LLMs ✅ [arXiv-2610.21244].

**What they claim.** Pick a bounded answer set up front, return a calibrated distribution over it in one forward pass, so code can
threshold, route and escalate; trade open-ended generation for type-safety, calibration and 1–2 orders of magnitude of latency/cost.
✅ [TS-blog-intro] [OAI-decisions] [MS-learn].

**How they are evaluated.** (i) the same typed questions put to LLMs, their answer-token probabilities read out (System1-Arena: 92k
questions / 105 tasks / 268 sources, plus 138 closed-loop environments across web shopping, SWE-bench control, robotics, games,
driving) ✅; (ii) frozen-template multi-dataset harnesses ✅ [arXiv-2609.37647]; (iii) calibration as a first-class metric (Brier/ECE)
✅; (iv) cost and p50 latency per decision, and cost-per-successful-task for SLM/LLM cascades ✅ [arXiv-2510.03847]; (v) content-preserving
input rewrites to test robustness ✅ [arXiv-2610.19536].

**Headline results worth keeping.**
- Largest System-1 models match **fast** closed LLMs on static decisions at ~1/4 the latency, with better-calibrated probabilities;
  a reasoning LLM is the most accurate and costs ~70x a fast LLM (Claude Opus 5.5: 81.1% at $10.38/1k vs Jev 72.6% at $0.052/1k) ✅.
- **Agentic performance is the weak axis**: step accuracy does not predict loop success; small System-1 models reach ≤3.1% success on
  long-horizon tasks; re-training an LLM into a decision interface (5 paired comparisons) lowered agentic step accuracy in all five ✅.
- Typed decision models are as accurate as LLMs on tabular decisions at lower cost and less sensitive to how the question is worded, but
  no model was robust to how the table was organized, and ≥1/3 of row decisions flipped under some rewrite ✅ [arXiv-2610.19536].
- **Cost scales with the candidate menu**: Certo (a 4B non-generative action scorer) is accurate as a joint scorer (state+rules+candidate
  read together) but a cacheable independent encoder — ~5x cheaper at 77 candidates — lost rule sensitivity (recall@1 1.00→0.24), and a
  shortlist+rerank rescue failed; counterfactual supervision recovered held-out synthetic rules but not (established) real ones ✅ [arXiv-2609.37832].

**Related work on fast–slow / enumerated-action selection.**
- **Set-of-Mark (SoM)** overlays numbered regions on an image and has the VLM refer to them by index — the interface our marks already
  implement, and the standard trick for making a VLM choose among visual options ✅ [arXiv-2310.11441].
- **SwiftSage** — small fine-tuned LM picks the next action, LLM does subgoal planning/grounding; the canonical two-module fast/slow agent ✅
  [arXiv-2305.17390]. **FaST** ("Visual Agents as Fast and Slow Thinkers") adds a switch adapter between System-1/System-2 paths ✅
  [arXiv-2408.08862]. **Reasoning on a Spectrum** aligns one LLM to S1 vs S2 styles and measures the accuracy/efficiency trade-off ✅ [arXiv-2502.12470].
- **Robotics dual-system VLA**: a slow VLM planner paired with a fast policy/action expert running at control rate — the "fast action
  model" pattern in embodied settings, with the same latency argument. OpenHelix surveys the structural variants of dual-system designs,
  Hume adds System-2-style reasoning on top of a VLA, and π0 is a flow-matching VLA built on a pretrained VLM ✅ [arXiv-2505.03912]
  [arXiv-2505.21432] [arXiv-2410.24164].
- **SLM-default / LLM-fallback** architectures with uncertainty-aware routing and verifier cascades; metrics are cost-per-success, schema
  validity, executable-call rate, p50/p95 latency ✅ [arXiv-2510.03847].
- System1-Arena's action encodings are the exact "enumerated choice" pattern: pick one of 43 tools or "done", then each argument from
  candidates drawn from the tool schema; or pick one legal chess move ✅ [arXiv-2610.21244].

---

## 3. Fit for our harness (numbered marks, enumerated tools, C1..Cn accept/reject)

**Structurally, the match is good.** Our action space is already closed and enumerable, which is the one thing decision models are built
for: Jev's docs recommend converting bounded extraction into a Choice over the options ("when the answer space is bounded, turn extraction
into a Choice over the options"), accept up to 255 options, and its jaggedness page points the same way. Single-step tool/candidate choice
is where System-1 models match LLMs, and cost/latency (~$0.052/1k requests, 321 ms measured) is negligible next to our planner runs
(≈$0.04 medium / $0.25 xhigh API-equivalent per case, minutes of wall-clock, E9/E11). ✅ [TS-docs-choice] [TS-docs-jagged] [arXiv-2610.21244].

**There is a quantified precedent of our exact use-case class.** TypeSafe's "skill suggestion" cookbook picks **at most one of 182 skills**
for an agent turn with two Jev requests (rank, then re-check the top 3), and over 488 requests against `claude-haiku-4-5` cut wrong-skill
loads from 16.8% to 7.3% and "loaded one when nothing fits" from 9.8% to 4.0% (floor with the right answer handed over: 2.5%/1.2%) — the
model ranks, the agent keeps its own judgement, code owns the roster. The same docs score 218 enumerated line ids against a query in one
request ✅ [TS-cookbook-skill] [TS-cookbook-linebyline]. This is "slow planner + fast selector over enumerated options", already measured.

**What is known vs speculation.**
- *Known:* enumerated Choice/Noul over ≤255 options with calibrated probabilities and confidence; 70–500 ms (vendor) / 321 ms
  (independent); $0.042/Mtok (vendor) / $0.052 per 1k requests (independent); parallel fan-out of many questions in one request at
  ~constant latency; confidence-gated escalation to "a person or a more expensive reasoning model" as a documented pattern; agentic
  closed-loop performance is the weak axis (17.4%) and should not own the loop. ✅
- *Known and adverse to us:* image/Box paths are behind a danger-zone header, vendor-labelled non-production, and acknowledged to be
  less accurate than text; option-order bias toward the first option; unreliable counting/arithmetic/date comparison; accuracy degrades
  as the state fills with irrelevant detail ("context rot"). ✅ [TS-docs-images] [TS-docs-jagged].
- **[speculation]** Whether a fast selector helps *our* loop at all. Nothing published evaluates Jev (or any decision model) on medical
  images, on rendered numbered marks, or on interactive 3D segmentation — searched, ❓. Sending a slice render as `state` plus "which mark
  lies on the lesion's inferior margin?" is precisely the SoM-style query that has no evidence base for Jev. The text-only claim ("is
  candidate C7 inside tumour tissue, given these numbers?") is much better supported than the pixel claim.
- **[speculation]** Integration sketch, untested: keep the slow LLM as planner/driver (it owns the loop, `seg_view`/`seg_finish`, and
  writes nothing to a decision model); the harness computes the bounded candidate set per step (enumerated tools, marks in the current
  view, C1..Cn) and asks Jev **one** request containing one `choice` over the ids plus supporting `noul`s ("is C7 inside tumour tissue?",
  "does C7 overlap the current mask?"), with code thresholds using `probabilities`/`confidence` to act or escalate to the planner. Ask the
  accept/reject Noul for every C1..Cn in that same request (fan-out is nearly latency-free, but each extra question costs tokens). Never
  ask it to count marks, add areas or compare Dice — arithmetic stays in code. Shuffle option order to blunt the first-option bias.
- **Cheapest useful experiment [speculation].** Replay the recorded trajectories (`results/*/sessions`) as decision requests — "which of
  the N enumerated options did the agent pick?" — and score Jev against the agent's actual choice: text-only candidate lists first, then a
  second arm with the rendered slice attached. No harness change; it measures the selector before anyone builds a bridge. Also note that
  image limits (280–1568 px per side) mean our multi-panel sheets need re-rendering before they can be sent at all.

---

## Sources (all fetched on 2026-10-19)

| Tag | Source |
| --- | --- |
| ✅ [TS-home] | TypeSafe AI homepage — pricing, 70–500 ms / 193.6x–444.6x claims, "System One Models" definition — https://typesafe.ai/ |
| ✅ [TS-blog-intro] | "Introducing System One Models & Jev" (2026-09-15) — RLCD, parallel sampler, 70–500 ms, $0.042/Mtok, workflow evals — https://typesafe.ai/blog/introducing-system-one-models-and-jev |
| ✅ [TS-blog-1.14] | "Jev: For the builders AND dreamers" (2026-10-14) — images, Box, long context, jev-1.14.0, danger-zone policy — https://typesafe.ai/blog/jev-1-14 |
| ✅ [TS-docs-intro] | Docs — Introduction (state + typed questions, side-effect-free primitives) — https://docs.typesafe.ai/introduction.md |
| ✅ [TS-docs-models] | Docs — Models: `jev-1.14.0`/`jev-1.13.0`, $42/Btok, context 64k/32k (262k danger zone), rate limits, aliases, text-only statement — https://docs.typesafe.ai/models.md |
| ✅ [TS-docs-choice] | Docs — Choice: ≤255 options, probabilities, confidence, fan-out guidance — https://docs.typesafe.ai/primitives/choice.md |
| ✅ [TS-docs-images] | Docs — Images: composable assets, "Dreamer-ready" callout, 280–1568 px / 16 MiB limits, image tokens — https://docs.typesafe.ai/images.md |
| ✅ [TS-docs-box] | Docs — Box question type (jev-1.14 only), bounding box as fractions — https://docs.typesafe.ai/primitives/box.md |
| ✅ [TS-docs-jagged] | Docs — "Jev 1.13 jaggedness": literal reading, counting, dates, indirection, large state, adversarial state, option-order bias — https://docs.typesafe.ai/model-jaggedness/jev-1.13.md |
| ✅ [TS-cookbook-skill] | Cookbook — Skill suggestion: pick ≤1 of 182 skills, 16.8%→7.3% wrong-skill loads over 488 requests — https://docs.typesafe.ai/cookbooks/skill_suggestion.md |
| ✅ [TS-cookbook-linebyline] | Cookbook — Line-by-line search: 218 enumerated line ids scored in one request — https://docs.typesafe.ai/cookbooks/semantic_find.md |
| ✅ [arXiv-2609.37647] | Deußer, Sparrenberg, Sifa, "Evaluating and Benchmarking the System One Model Jev", 2026-09-29 — 37 datasets, 346,009 requests, <$10 — https://arxiv.org/abs/2609.37647 |
| ✅ [arXiv-2610.21244] | Chen, Liu, An, Wang, Zhou, "System1-Arena: Which Decisions Can Be Left to System 1?", 2026-10-15 — 30 models, 92k questions, 138 closed-loop envs, latency/cost, action encodings — https://arxiv.org/abs/2610.21244 |
| ✅ [arXiv-2610.19536] | "Typed Decision Models on Tabular Data", 2026-10-14 — 4 decision models vs 3 LLMs, 21 rewrites — https://arxiv.org/abs/2610.19536 |
| ✅ [arXiv-2609.37832] | "Can a Cacheable Decision Model Follow Rules?" (Certo, Qwen3-4B), 2026-09-29 — menu-size cost, joint vs cacheable scorer — https://arxiv.org/abs/2609.37832 |
| ✅ [OAI-decisions] | OpenAI API docs — Decisions (beta, `POST /v1/decisions`, `gpt-6-luna`, text **or images**, predicate/choice/score) — https://developers.openai.com/api/docs/guides/decisions.md |
| ✅ [OAI-changelog] | OpenAI API changelog — Decisions API beta entry under **Oct 6, 2026** ("10x faster than the Responses API") — https://developers.openai.com/api/docs/changelog.md |
| ✅ [OAI-pricing] | OpenAI pricing — `gpt-6-luna` $0.10 per 1M short-context input tokens (standard tier) — https://developers.openai.com/api/docs/pricing.md |
| ✅ [MS-learn] | Microsoft Learn — "Deploy and use Microsoft-Decision-1 in Microsoft Foundry": noul/choice/score over text-or-JSON state, routing pattern, no image input claimed — https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision |
| ⚠️ [MS-blog] | commandline.microsoft.com announcement of Microsoft-Decision-1 — Cloudflare-blocked, not fetched; figures circulating elsewhere (85 ms p50, $0.042/Mtok) are **unverified**; the Qwen3.5-9B base model is from System1-Arena, not from this blog |
| ✅ [arXiv-2310.11441] | Set-of-Mark Prompting — https://arxiv.org/abs/2310.11441 |
| ✅ [arXiv-2305.17390] | SwiftSage, fast+slow generative agent — https://arxiv.org/abs/2305.17390 |
| ✅ [arXiv-2408.08862] | Visual Agents as Fast and Slow Thinkers (FaST) — https://arxiv.org/abs/2408.08862 |
| ✅ [arXiv-2502.12470] | Reasoning on a Spectrum: System 1 / System 2 alignment — https://arxiv.org/abs/2502.12470 |
| ✅ [arXiv-2505.03912] | OpenHelix: dual-system VLA survey + open model — https://arxiv.org/abs/2505.03912 |
| ✅ [arXiv-2505.21432] | Hume: System-2 thinking in a VLA model — https://arxiv.org/abs/2505.21432 |
| ✅ [arXiv-2410.24164] | π0: VLA flow model for general robot control — https://arxiv.org/abs/2410.24164 |
| ✅ [arXiv-2510.03847] | Small Language Models for Agentic Systems: survey (SLM-default/LLM-fallback, cost-per-success) — https://arxiv.org/abs/2510.03847 |

**Explicit gaps (❓).** No vendor paper or model card; architecture/data/parameter count undisclosed; no source evaluating Jev or any
decision model on medical images, rendered numbered marks, or interactive 3D segmentation; image-token cost per render unpublished;
JevBench and ReflexBench (third-party decision benchmarks cited by System1-Arena) not fetched; a search for tooling returned no
decision-model bridge for `nnInteractive` or the pi harness.
