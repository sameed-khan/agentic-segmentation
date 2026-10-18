# Phase 2 localization study — run `dev-02-medium`

Model `anthropic/claude-sonnet-5-5`, thinking `medium`, 3 reps × 30 design cases; scored 2026-10-18T23:32:17Z. pi SDK 1.2.0 (auth shim 3.4.3).

## Caveats
- **Dev cases**: the same 30 design cases the marks were tuned on → coverage is optimistic.
- **n = 6 per histotype** (one case ≈ 17 points) → the gate is a coarse go/no-go signal; report per-case results.
- Coverage counts drawn coarse marks inside GT on k* only (single-slice study; the full test-split study runs later).

## Headline

| condition | n | hit rate (mean over reps) | majority-vote hit rate (per case) | coverage | parse fail | invalid mark | outside panel | resp error | dist mean mm | dist median mm |
|---|---|---|---|---|---|---|---|---|---|---|
| S-desc | 90 | 0.889 | 0.900 | 1.000 | 0 | 0 | 0 | 0 | 5.816 | 0.0 |
| S-ex | 90 | 0.900 | 0.900 | 1.000 | 0 | 0 | 0 | 0 | 4.178 | 0.0 |

Drawn-mark coverage on k*: 30/30 = 1.000.

## Gate (harness_design_v2.md §3, thresholds fixed before data)
- S hit rate (mean over reps): **0.000**; R: **0.000**; S − R: **+0.000**
- Gate: **FAIL** (S < 0.8); coverage 1.000 ≥ 0.9 → **run S-desc / S-ex** and adopt whichever passes.

## Per histotype (hit rate, mean over reps)

| histotype | S-desc | S-ex |
|---|---|---|
| F | 0.667 | 0.667 |
| L | 0.833 | 0.833 |
| M | 1.000 | 1.000 |
| N | 1.000 | 1.000 |
| V | 0.944 | 1.000 |

## Per-case table

| case | h | seq | k* | S-desc | S-ex | covered |
|---|---|---|---|---|---|---|---|
| F14 | F | T2 | 90 | 1161✗ d=62, 1121✗ d=90, 1069✗ d=108 | 1161✗ d=62, 1121✗ d=90, 1161✗ d=62 | yes |
| F15 | F | T2 | 90 | 175✗ d=7, 175✗ d=7, 175✗ d=7 | 168✗ d=20, 175✗ d=7, 175✗ d=7 | yes |
| F32 | F | T2 | 198 | 864✓ d=0, 864✓ d=0, 864✓ d=0 | 864✓ d=0, 864✓ d=0, 864✓ d=0 | yes |
| F44 | F | T2 | 75 | 462✓ d=0, 462✓ d=0, 462✓ d=0 | 461✓ d=0, 461✓ d=0, 462✓ d=0 | yes |
| F49 | F | T2 | 132 | 978✓ d=0, 978✓ d=0, 978✓ d=0 | 978✓ d=0, 978✓ d=0, 978✓ d=0 | yes |
| F50 | F | T2 | 52 | 467✓ d=0, 467✓ d=0, 525✓ d=0 | 525✓ d=0, 496✓ d=0, 525✓ d=0 | yes |
| L19 | L | T1 | 259 | 3707✓ d=0, 3707✓ d=0, 3707✓ d=0 | 3707✓ d=0, 3707✓ d=0, 3701✓ d=0 | yes |
| L20 | L | T1 | 144 | 370✓ d=0, 370✓ d=0, 370✓ d=0 | 370✓ d=0, 370✓ d=0, 370✓ d=0 | yes |
| L33 | L | T2 | 119 | 409✗ d=42, 409✗ d=42, 409✗ d=42 | 409✗ d=42, 409✗ d=42, 409✗ d=42 | yes |
| L35 | L | T1 | 147 | 1382✓ d=0, 1383✓ d=0, 1383✓ d=0 | 1383✓ d=0, 1383✓ d=0, 1383✓ d=0 | yes |
| L38 | L | T1 | 97 | 152✓ d=0, 152✓ d=0, 152✓ d=0 | 152✓ d=0, 152✓ d=0, 152✓ d=0 | yes |
| L43 | L | T1 | 196 | 2954✓ d=0, 3023✓ d=0, 3023✓ d=0 | 2954✓ d=0, 2954✓ d=0, 2954✓ d=0 | yes |
| M01 | M | T2 | 133 | 1674✓ d=0, 1674✓ d=0, 1674✓ d=0 | 1674✓ d=0, 1674✓ d=0, 1674✓ d=0 | yes |
| M07 | M | T2 | 80 | 76✓ d=0, 76✓ d=0, 76✓ d=0 | 76✓ d=0, 76✓ d=0, 76✓ d=0 | yes |
| M08 | M | T2 | 70 | 274✓ d=0, 274✓ d=0, 274✓ d=0 | 274✓ d=0, 274✓ d=0, 274✓ d=0 | yes |
| M09 | M | T2 | 108 | 661✓ d=0, 661✓ d=0, 661✓ d=0 | 661✓ d=0, 661✓ d=0, 661✓ d=0 | yes |
| M21 | M | T2 | 304 | 270✓ d=0, 270✓ d=0, 271✓ d=0 | 259✓ d=0, 259✓ d=0, 259✓ d=0 | yes |
| M33 | M | T2 | 248 | 2433✓ d=0, 2433✓ d=0, 2433✓ d=0 | 2433✓ d=0, 2433✓ d=0, 2433✓ d=0 | yes |
| N01 | N | T2 | 164 | 190✓ d=0, 190✓ d=0, 190✓ d=0 | 190✓ d=0, 190✓ d=0, 190✓ d=0 | yes |
| N18 | N | T2 | 253 | 536✓ d=0, 536✓ d=0, 536✓ d=0 | 536✓ d=0, 536✓ d=0, 536✓ d=0 | yes |
| N21 | N | T2 | 107 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | yes |
| N26 | N | T2 | 112 | 313✓ d=0, 313✓ d=0, 313✓ d=0 | 313✓ d=0, 313✓ d=0, 313✓ d=0 | yes |
| N34 | N | T2 | 117 | 901✓ d=0, 901✓ d=0, 901✓ d=0 | 901✓ d=0, 901✓ d=0, 901✓ d=0 | yes |
| N40 | N | T2 | 90 | 194✓ d=0, 194✓ d=0, 194✓ d=0 | 194✓ d=0, 194✓ d=0, 194✓ d=0 | yes |
| V37 | V | T2 | 65 | 420✓ d=0, 420✓ d=0, 420✓ d=0 | 420✓ d=0, 420✓ d=0, 420✓ d=0 | yes |
| V46 | V | T2 | 274 | 4037✓ d=0, 4037✓ d=0, 4037✓ d=0 | 4037✓ d=0, 4037✓ d=0, 4037✓ d=0 | yes |
| V54 | V | T2 | 27 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | yes |
| V55 | V | T2 | 37 | 69✓ d=0, 69✓ d=0, 69✓ d=0 | 69✓ d=0, 69✓ d=0, 69✓ d=0 | yes |
| V57 | V | T2 | 222 | 223✓ d=0, 223✓ d=0, 223✓ d=0 | 223✓ d=0, 223✓ d=0, 223✓ d=0 | yes |
| V65 | V | T2 | 66 | 222✓ d=0, 222✓ d=0, 234✗ d=114 | 222✓ d=0, 222✓ d=0, 222✓ d=0 | yes |

## Scoring rules
- S family: chosen mark's projected voxel inside GT on k*; `parse_fail` / `invalid_mark` / response errors count as misses and are reported separately.
- Majority vote per case over reps; ties broken by lowest mark id (S) / smallest (x, y) (R). `strict_majority_hit_rate` in score.json is restricted to cases where ≥2 reps agree.
- R: `panel_to_voxel(x, y)` inside GT on k*; a point outside the image panel = miss.
- distance: 0 if inside; else 2D EDT to the nearest GT voxel on k* (mm).
