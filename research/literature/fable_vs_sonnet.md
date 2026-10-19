# Fable 5 / Fable 5.1 vs Sonnet 5.5 — web research for our MRI set-of-marks segmentation agent

Method: `curl` + DuckDuckGo/arXiv HTML search; Anthropic system cards fetched as PDFs and text-extracted; every factual
claim below is tied to a fetched source. `⚠UNVERIFIED` marks anything I could not confirm from a fetched page.
Our local `pi` provider catalog (`…/pi-ai/dist/providers/data/anthropic.json`) is used for exact ids/pricing/limits.

## Bottom line
- Fable 5 / 5.1 sit **above Opus in Anthropic's hierarchy** ("Mythos-class"), but on the one **same-harness public
  head-to-head** the Sonnet 5.5 system card reports, **Sonnet 5.5 ≥ Fable 5.1 on nearly every agentic, multimodal and
  clinical benchmark** (Terminal-Bench, Toolathlon, GDPval, AA-Briefcase, Chartography, BenchCAD, HealthBench,
  HealthBench Professional, PhysicianBench, ArXivMath). Fable 5.1 only leads on knowledge/reasoning (HLE no-tools,
  GMMLU/MILU, ARC-AGI, GPQA).
- **No public benchmark measures our task** (set-of-marks pointing on MRI + iterative segmentation + mask self-judgement).
  There is **no evidence** Fable 5/5.1 beats Sonnet 5.5 on it; the closest proxies lean slightly toward Sonnet 5.5.
- Fable costs ~5–7× more, is ~2× slower, and carries biology/cyber **safeguard fallbacks** that produced large refusal
  rates on biomedical questions in an independent study (Fable 5). A short, effort-matched pilot is the only way to know.

## 1. What Fable 5 / Fable 5.1 are
- **Tier/positioning**: Anthropic's model nav orders **Mythos → Fable → Opus → Sonnet → Haiku**
  (anthropic.com; footer + `anthropic.com/claude/fable`). "Mythos-class models … sit above our Opus class in
  capability" (`anthropic.com/news/claude-fable-5-mythos-5`, Jun 9 2026). Fable = Mythos with stronger safeguards;
  same underlying model (Fable 5.1 & Mythos 5.1 page, Sep 2026).
- **Ids / dates**: `claude-fable-5` launched **Jun 9 2026**; access **suspended Jun 12 → restored Jul 1 2026** (US
  export controls on the model) (`…/news/redeploying-fable-5`). `claude-fable-5-1` announced **Sep 2026** (system card
  / AA list release date **2026-09-01**). `claude-sonnet-5-5` launched **Sep 28 2026** (`…/news/claude-sonnet-5-5`).
  `claude-opus-5-5` launched Sep 22 2026 and is billed as "perform[ing] at the level of Claude Fable 5.1 on most work".
- **Catalog facts** (pi `anthropic.json`): Fable 5 & 5.1 — 1M ctx, 128K out, text+image, reasoning, effort map
  `off:null, low…max`; `forceAdaptiveThinking`. Sonnet 5.5 — 1M/128K, text+image, reasoning, same effort map.
- **Pricing** (pi catalog, matches launch posts): Fable **$10/$50** per Mtok in/out, cache-write $12.50; Fable 5.1
  cache-read **$0.25** (5.1 reduced cache reads 75%; Fable 5 was $1.00). Sonnet 5.5 **$2/$10**, cache-write $2.50,
  cache-read **$0.10**.
- **Fable 5.1 positioning**: "world's most advanced model for coding and knowledge work", long-running /
  long-horizon, self-verification; defaults High effort in Claude Code, Medium on claude.ai/Cowork
  (`anthropic.com/claude-fable-and-mythos-5-1`).

## 2. Official head-to-head: Sonnet 5.5 vs Fable 5.1 (same internal harness, max effort)
All from `anthropic.com/claude-sonnet-5-5-system-card` §8 unless noted; Fable-side values are the ones the Sonnet card
itself reports (best available controlled comparison).
| Evaluation (domain) | Sonnet 5.5 | Fable 5.1 | Notes |
|---|---|---|---|
| Chartography, no tools / tools (vision) | **61.6 / 90.2** | 44.8 / 88.4 | Sonnet clearly better raw vision |
| BenchCAD Vision2Code IoU, no tools / tools (vision→code) | **0.747 / 0.963** | 0.606 / 0.926 | Sonnet better |
| Terminal-Bench 4.0 (agentic terminal) | **70.6** (blog) | 56.0–55.8 | Sonnet better |
| Terminal-Bench-Science 0.1 | **59.9** | 52.6 | Sonnet better |
| Toolathlon-Veriﬁed Pass@1 / @3 / @³ / avg turns | **77.8 / 85.2 / 68.5 / 31.6** | 77.8 / 81.5 / 73.1 / **23.7** | tie@1; Fable more consistent, fewer turns |
| GDPval-AA v2.1 Elo | **1844** | 1735 | Sonnet better |
| AA-Briefcase v1.1 Elo | **1811** | 1678 | Sonnet better |
| Humanity's Last Exam no-tools | 56.9 | **60.9** | Fable better (knowledge) |
| GMMLU / MILU | 92.1 / 91.6 | **94.0 / 93.0** | Fable better (knowledge/multilingual) |
| ArXivMath no tools / tools | **86.8 / 95.2** | 82.9 / 92.1 | Sonnet better |
- Fable 5.1 system-card summary table (own card, different column set) independently self-reports OSWorld 2.0
  77.9/41.7 (partial/strict), Terminal-Bench 4.0 56% (61%), AutomationBench 31.4, GDPval-AA 1853 — **different
  benchmark versions than Sonnet 5.5's (OSWorld 2.1), so those two rows are not comparable.**
- Fable 5 card (`anthropic.com/claude-fable-5-mythos-5-system-card`) reports vision evals (ScreenSpot-Pro 87.3/90.7
  for Mythos 5; CharXiv, ChartQAPro) against **Sonnet 4.6 / Opus 4.8 — never Sonnet 5.5.** ⚠ No Fable-vs-Sonnet-5.5
  ScreenSpot/pointing number exists in either card.

## 3. Independent evaluations
- **Artificial Analysis** (`/models/releases/comparisons/claude-sonnet-5-5-vs-claude-fable-5-1`; `/models/comparisons/claude-sonnet-5-5-xhigh-vs-claude-fable-5-1`):
  Intelligence Index Sonnet 5.5 (max) **56** vs Fable 5.1 (max) **53**; at xhigh/max pairing 52 vs 53. Fable 5.1
  better on HLE (59 vs 50), AA-Omniscience (43 vs 24), SciCode (63 vs 57), AA-LCR (85 vs 80). Sonnet 5.5 better on
  **AutomationBench 66 vs 59, Terminal-Bench 4.0 57 vs 52, GDPval-AA 1754 vs 1728.** Healthcare & Medical index:
  **58 vs 58 (tie)**. Cost/task $2.01 vs $7.63; output 112 vs 68 tok/s; TTFT 36.2s vs 141.6s.
- **AA MLCR-AA** (medical long-context reasoning, `/evaluations/mlcr-aa`): **Sonnet 5.5 (max) 75.0% > Fable 5.1 71.1%
  > Opus 5.5 66.7%.**
- **BenchLM** (`benchlm.ai/compare/claude-fable-5-1-vs-claude-sonnet-5-5`): Agentic composite **Fable 5.1 80.9 (#2/128)
  vs Sonnet 5.5 69.3 (#11/128)**. Coding composite: **Sonnet 5.5 84.2 (#1/150) vs Fable 5.1 76.8 (#3/150)**.
  Multimodal category: **Sonnet 5.5
  86.7, Fable 5.1 "Not ranked / not comparable"** (Fable has no weighted vision rows).
- **LLM-Stats** (`llm-stats.com/models/compare/claude-fable-5-1-vs-claude-sonnet-5-5`): Vision 40.9 vs **41.5**,
  Tool-use 27.0 vs **35.4** (Sonnet higher); Sonnet wins 3/3 shared benchmarks.
- **Vals** (`vals.ai/benchmarks/…`): MMMU Pro — **Fable 5.1 #1 at 90.64%** (Sonnet 5.5 not yet listed; page dated
  9/1/2026). MedScribe — Opus 5.5 91.43 > Fable 5.1 91.29 > Sonnet 5.5 91.10 (near-tie). BioMysteryBench — **Sonnet
  5.5 #1 at 81.11%**.
- LMArena vision: ⚠ could not fetch (arena.ai returns HTTP 403 to curl); no verified LMArena-vision numbers.

## 4. Medical / imaging evidence
- **Sonnet 5.5 card**: Biomedical image analysis (CT, **MRI**, histopath, dermoscopy, retinal, US, microscopy) —
  Sonnet 5.5 **72.2%**, Opus 5.5 71.4, GPT-6 Astra 77.5. **Fable 5.1 is not reported on this benchmark.**
- **Healthcare head-to-heads (Sonnet card §8.15)**: HealthBench Professional raw **Sonnet 5.5 77.1 ≈ Opus 5.5 77.1 >
  Fable 5.1 74.2**; length-adjusted **Sonnet 69.2 > Opus 5.5 65.6 > Fable 5.1 62.1**. HealthBench: Sonnet 5.5 higher
  than Fable 5.1 at every effort after adjustment. **PhysicianBench (EHR agentic clinical tasks): Sonnet 5.5 63.2% ≈
  Fable 5.1 61.0%** (> Opus 5 57.6, < Opus 5.5 68.4).
- **Safeguard/refusal risk (Fable 5 only)**: independent arXiv:2607.10849 ("Capabilities of Claude Fable 5 on
  Biomedical Challenge Problems") finds Fable 5 **refuses 8.0%–99.4% of biomedical questions** across 8 benchmarks
  (refusals tracked as a distinct outcome); once refusals are excluded its accuracy ≥ all baselines. Anthropic's Aug 7
  2026 post says Fable 5.1's biology classifiers fire **~85% less** on benign medical/biology queries, but life-science
  R&D queries are still redirected to Opus models. ⚠ No equivalent refusal-rate study exists for Fable 5.1.
- arXiv:2606.18193 (red-team, Fable 5/5.1/Opus 4.8) is safety-only; arXiv radiology-VLM surveys (2608.13309 MRI
  longitudinal, 2608.21140 CT spatial agent) do **not** evaluate Fable vs Sonnet 5.5. No medical-imaging VLM paper
  comparing these two models was found.

## 5. Agent persistence / iteration / verbosity
- No independent head-to-head on "lazy / first-answer-acceptance" exists. Same-harness Toolathlon avg turns:
  **Sonnet 5.5 31.6 vs Fable 5.1 23.7** — i.e., Sonnet took *more* turns, not fewer (Sonnet card Table 8.14.5.A).
- Official framing favours Fable on long-horizon persistence: "the longer and more complex the task, the larger Fable
  5's lead"; partner quotes cite self-verification, multi-hour/38-hour unattended runs, "keeps its own records,
  reprioritizes" (Fable 5.1 page). These are vendor/partner claims, not controlled comparisons.
- Sonnet 5.5 is repeatedly described as moving in the *same* direction: Base44 measured **3.6 iterations/build vs 7.7
  for Opus 5**, "fewest failed tool calls", "rarely stopped mid-build to ask" (Sonnet 5.5 blog).
- Sonnet 5.5 card §8.15.4 is directly relevant: for **agentic EHR clinical tasks effort matters a great deal**
  (PhysicianBench low 27.2% → max 63.2%, ~14.5 min/task at max), while static health QA barely moves with effort.

## 6. Cost / latency / availability
- ~5× per-token cost and ~2× latency for Fable (see §1, §3). AA time/task 737.9s vs 424.0s; end-to-end 148.9s vs 40.6s.
- Availability: Fable 5 was pulled for ~3 weeks by US export control (Jun 12–Jul 1 2026); Fable usage on subscription
  plans is metered/credit-based (launch + redeploy posts). Both Fable models apply **biology/cyber classifier fallbacks
  to Opus** — for a medical-imaging agent this can silently swap the model mid-run.
- ⚠ Rate-limit tables not fetched (docs pages not reached); treat exact RPM/TPM as unknown.

## 7. Known vs unknown, and a minimal fair comparison
**Known:** Fable 5/5.1 are a top-tier (above-Opus) family, 1M context, vision-capable, high effort on long-horizon
agentic work; on the only same-harness head-to-heads they are **not** better than Sonnet 5.5 for agentic tool-use,
chart/visual reasoning, vision→code, or clinical tasks, and Sonnet 5.5 is cheaper/faster. Fable 5.1 leads on
knowledge/reasoning benchmarks only.
**Unknown for our task:** set-of-marks pointing on MRI, iterative segmentation refinement behaviour, and mask-correctness
judgement have **no public Fable-vs-Sonnet-5.5 numbers**; Fable 5.1's biomedical refusal rate under real medical-imaging
prompts is unmeasured; whether Fable's long-horizon self-verification translates into more/better `seg_*` refinement
calls is untested.
**Minimal fair comparison** (cheap, ~1 day):
1. Same harness commit (e.g. `harness-v2-rc1`) and prompts; models `anthropic/claude-sonnet-5-5` vs
   `anthropic/claude-fable-5-1` (and optionally `claude-fable-5`).
2. Match effort: run both at the same explicit level (high, then xhigh) **and** each at max, since Sonnet's agentic
   advantage is effort-dependent.
3. Same 30-case phase-1 set, ≥3 reps each; log per-case Dice, localization-failure flag, tool calls, model prompts,
   and a computed "refined after first result?" bit.
4. Re-run the single-slice set-of-marks pointing study (phase-2 S condition) for both models to isolate pointing skill
   from agent behaviour.
5. Instrument refusals/fallbacks (Fable classifier) and cost/latency; only then treat a Dice/localization delta as real.

Sources (all fetched): anthropic.com (`/news/claude-fable-5-mythos-5`, `/news/redeploying-fable-5`,
`/news/improving-fable-5-s-biology-safeguards`, `/claude-fable-and-mythos-5-1`, `/news/claude-sonnet-5-5`);
system cards `claude-fable-5-mythos-5-system-card`, `claude-fable-5-1-mythos-5-1-system-card`,
`claude-sonnet-5-5-system-card`; artificialanalysis.ai (release + model comparison, `evaluations/mlcr-aa`,
`evaluations/mmmu-pro`, leaderboards); benchlm.ai; llm-stats.com; vals.ai; arXiv 2607.10849, 2606.18193, 2608.13309,
2608.21140; local pi catalog `anthropic.json`.
