# Phase 2 localization study — run `smoke2`

Model `anthropic/claude-sonnet-5-5`, thinking `medium`, 1 reps × 30 design cases; scored 2026-10-18T18:46:20Z. pi SDK 1.3.0 (auth shim 3.4.3).

## Caveats
- **Dev cases**: the same 30 design cases the marks were tuned on → coverage is optimistic.
- **n = 6 per histotype** (one case ≈ 17 points) → the gate is a coarse go/no-go signal; report per-case results.
- Coverage counts drawn coarse marks inside GT on k* only (single-slice study; the full test-split study runs later).

## Headline

| condition | n | hit rate (mean over reps) | majority-vote hit rate (per case) | coverage | parse fail | invalid mark | outside panel | resp error | dist mean mm | dist median mm |
|---|---|---|---|---|---|---|---|---|---|---|
| R | 1 | 0.000 | 0.000 | – | 0 | 0 | 0 | 0 | 81.939 | 81.939 |
| S | 1 | 0.000 | 0.000 | 0.867 | 0 | 0 | 0 | 0 | 75.432 | 75.432 |
| S-desc | 1 | 0.000 | 0.000 | 0.867 | 0 | 0 | 0 | 0 | 75.432 | 75.432 |
| S-ex | 1 | 0.000 | 0.000 | 0.867 | 0 | 0 | 0 | 0 | 75.432 | 75.432 |

Drawn-mark coverage on k*: 26/30 = 0.867; cases without a drawn mark inside GT on k*: N01, N21, V37, V57.

## Gate (harness_design_v2.md §3, thresholds fixed before data)
- S hit rate (mean over reps): **0.000**; R: **0.000**; S − R: **+0.000**
- Gate: **FAIL** (S < 0.8) and drawn-mark coverage 0.867 < 0.9 → **fix marks** (design cases only), then rerun.

## Per histotype (hit rate, mean over reps)

| histotype | R | S | S-desc | S-ex |
|---|---|---|---|---|
| F | 0.000 | 0.000 | 0.000 | 0.000 |
| L | – | – | – | – |
| M | – | – | – | – |
| N | – | – | – | – |
| V | – | – | – | – |

## Per-case table

| case | h | seq | k* | R | S | S-desc | S-ex | covered |
|---|---|---|---|---|---|---|---|---|---|
| F14 | F | T2 | 90 | (315,245)✗ d=82 | 478✗ d=75 | 478✗ d=75 | 478✗ d=75 | yes |
| F15 | F | T2 | 90 | – | – | – | – | yes |
| F32 | F | T2 | 198 | – | – | – | – | yes |
| F44 | F | T2 | 75 | – | – | – | – | yes |
| F49 | F | T2 | 132 | – | – | – | – | yes |
| F50 | F | T2 | 52 | – | – | – | – | yes |
| L19 | L | T1 | 259 | – | – | – | – | yes |
| L20 | L | T1 | 144 | – | – | – | – | yes |
| L33 | L | T2 | 119 | – | – | – | – | yes |
| L35 | L | T1 | 147 | – | – | – | – | yes |
| L38 | L | T1 | 97 | – | – | – | – | yes |
| L43 | L | T1 | 196 | – | – | – | – | yes |
| M01 | M | T2 | 133 | – | – | – | – | yes |
| M07 | M | T2 | 80 | – | – | – | – | yes |
| M08 | M | T2 | 70 | – | – | – | – | yes |
| M09 | M | T2 | 108 | – | – | – | – | yes |
| M21 | M | T2 | 304 | – | – | – | – | yes |
| M33 | M | T2 | 248 | – | – | – | – | yes |
| N01 | N | T2 | 164 | – | – | – | – | **no** |
| N18 | N | T2 | 253 | – | – | – | – | yes |
| N21 | N | T2 | 107 | – | – | – | – | **no** |
| N26 | N | T2 | 112 | – | – | – | – | yes |
| N34 | N | T2 | 117 | – | – | – | – | yes |
| N40 | N | T2 | 90 | – | – | – | – | yes |
| V37 | V | T2 | 65 | – | – | – | – | **no** |
| V46 | V | T2 | 274 | – | – | – | – | yes |
| V54 | V | T2 | 27 | – | – | – | – | yes |
| V55 | V | T2 | 37 | – | – | – | – | yes |
| V57 | V | T2 | 222 | – | – | – | – | **no** |
| V65 | V | T2 | 66 | – | – | – | – | yes |

## Scoring rules
- S family: chosen mark's projected voxel inside GT on k*; `parse_fail` / `invalid_mark` / response errors count as misses and are reported separately.
- Majority vote per case over reps; ties broken by lowest mark id (S) / smallest (x, y) (R). `strict_majority_hit_rate` in score.json is restricted to cases where ≥2 reps agree.
- R: `panel_to_voxel(x, y)` inside GT on k*; a point outside the image panel = miss.
- distance: 0 if inside; else 2D EDT to the nearest GT voxel on k* (mm).
