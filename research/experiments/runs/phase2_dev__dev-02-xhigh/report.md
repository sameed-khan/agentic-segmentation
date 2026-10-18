# Phase 2 localization study — run `dev-02-xhigh`

Model `anthropic/claude-sonnet-5-5`, thinking `xhigh`, 3 reps × 30 design cases; scored 2026-10-18T23:32:20Z. pi SDK 1.2.0 (auth shim 3.4.3).

## Caveats
- **Dev cases**: the same 30 design cases the marks were tuned on → coverage is optimistic.
- **n = 6 per histotype** (one case ≈ 17 points) → the gate is a coarse go/no-go signal; report per-case results.
- Coverage counts drawn coarse marks inside GT on k* only (single-slice study; the full test-split study runs later).

## Headline

| condition | n | hit rate (mean over reps) | majority-vote hit rate (per case) | coverage | parse fail | invalid mark | outside panel | resp error | dist mean mm | dist median mm |
|---|---|---|---|---|---|---|---|---|---|---|
| R | 90 | 0.733 | 0.767 | – | 0 | 0 | 0 | 0 | 5.847 | 0.0 |
| S | 90 | 0.900 | 0.900 | 1.000 | 0 | 0 | 0 | 0 | 4.488 | 0.0 |
| S-ex | 90 | 0.900 | 0.900 | 1.000 | 0 | 0 | 0 | 0 | 4.344 | 0.0 |

Drawn-mark coverage on k*: 30/30 = 1.000.

## Gate (harness_design_v2.md §3, thresholds fixed before data)
- S hit rate (mean over reps): **0.900**; R: **0.733**; S − R: **+0.167**
- Gate (S ≥ 0.8 and S − R ≥ 0.1): **PASS** → proceed with S.

## Per histotype (hit rate, mean over reps)

| histotype | R | S | S-ex |
|---|---|---|---|
| F | 0.667 | 0.667 | 0.667 |
| L | 0.667 | 0.833 | 0.833 |
| M | 1.000 | 1.000 | 1.000 |
| N | 0.500 | 1.000 | 1.000 |
| V | 0.833 | 1.000 | 1.000 |

## Per-case table

| case | h | seq | k* | R | S | S-ex | covered |
|---|---|---|---|---|---|---|---|---|
| F14 | F | T2 | 90 | (310,248)✗ d=79, (350,268)✗ d=80, (350,268)✗ d=80 | 1121✗ d=90, 1121✗ d=90, 1161✗ d=62 | 1121✗ d=90, 1121✗ d=90, 1161✗ d=62 | yes |
| F15 | F | T2 | 90 | (395,385)✗ d=7, (265,395)✗ d=31, (275,400)✗ d=27 | 175✗ d=7, 175✗ d=7, 168✗ d=20 | 175✗ d=7, 175✗ d=7, 175✗ d=7 | yes |
| F32 | F | T2 | 198 | (540,425)✓ d=0, (535,420)✓ d=0, (535,430)✓ d=0 | 864✓ d=0, 864✓ d=0, 864✓ d=0 | 864✓ d=0, 864✓ d=0, 864✓ d=0 | yes |
| F44 | F | T2 | 75 | (425,365)✓ d=0, (420,365)✓ d=0, (420,365)✓ d=0 | 462✓ d=0, 462✓ d=0, 462✓ d=0 | 462✓ d=0, 462✓ d=0, 462✓ d=0 | yes |
| F49 | F | T2 | 132 | (475,455)✓ d=0, (475,455)✓ d=0, (475,453)✓ d=0 | 978✓ d=0, 978✓ d=0, 978✓ d=0 | 978✓ d=0, 978✓ d=0, 978✓ d=0 | yes |
| F50 | F | T2 | 52 | (365,580)✓ d=0, (360,585)✓ d=0, (365,585)✓ d=0 | 467✓ d=0, 525✓ d=0, 525✓ d=0 | 467✓ d=0, 525✓ d=0, 467✓ d=0 | yes |
| L19 | L | T1 | 259 | (365,570)✓ d=0, (360,572)✓ d=0, (365,575)✗ d=1 | 3707✓ d=0, 3707✓ d=0, 3707✓ d=0 | 3707✓ d=0, 3701✓ d=0, 3707✓ d=0 | yes |
| L20 | L | T1 | 144 | (485,345)✓ d=0, (485,345)✓ d=0, (485,345)✓ d=0 | 370✓ d=0, 370✓ d=0, 370✓ d=0 | 370✓ d=0, 370✓ d=0, 370✓ d=0 | yes |
| L33 | L | T2 | 119 | (400,375)✗ d=40, (395,370)✗ d=43, (395,370)✗ d=43 | 409✗ d=42, 409✗ d=42, 409✗ d=42 | 409✗ d=42, 409✗ d=42, 409✗ d=42 | yes |
| L35 | L | T1 | 147 | (480,550)✓ d=0, (460,540)✓ d=0, (480,540)✓ d=0 | 1383✓ d=0, 1383✓ d=0, 1383✓ d=0 | 1383✓ d=0, 1383✓ d=0, 1383✓ d=0 | yes |
| L38 | L | T1 | 97 | (388,425)✓ d=0, (385,425)✓ d=0, (385,428)✓ d=0 | 152✓ d=0, 152✓ d=0, 152✓ d=0 | 152✓ d=0, 152✓ d=0, 152✓ d=0 | yes |
| L43 | L | T1 | 196 | (487,295)✗ d=4, (480,288)✓ d=0, (485,290)✗ d=3 | 3023✓ d=0, 3023✓ d=0, 3023✓ d=0 | 2954✓ d=0, 3023✓ d=0, 3023✓ d=0 | yes |
| M01 | M | T2 | 133 | (254,493)✓ d=0, (254,493)✓ d=0, (253,493)✓ d=0 | 1674✓ d=0, 1674✓ d=0, 1674✓ d=0 | 1674✓ d=0, 1674✓ d=0, 1674✓ d=0 | yes |
| M07 | M | T2 | 80 | (435,552)✓ d=0, (435,553)✓ d=0, (435,552)✓ d=0 | 76✓ d=0, 76✓ d=0, 76✓ d=0 | 76✓ d=0, 76✓ d=0, 76✓ d=0 | yes |
| M08 | M | T2 | 70 | (340,380)✓ d=0, (340,380)✓ d=0, (340,380)✓ d=0 | 274✓ d=0, 274✓ d=0, 274✓ d=0 | 274✓ d=0, 274✓ d=0, 274✓ d=0 | yes |
| M09 | M | T2 | 108 | (163,379)✓ d=0, (163,379)✓ d=0, (163,380)✓ d=0 | 661✓ d=0, 661✓ d=0, 661✓ d=0 | 661✓ d=0, 661✓ d=0, 661✓ d=0 | yes |
| M21 | M | T2 | 304 | (300,355)✓ d=0, (295,350)✓ d=0, (300,355)✓ d=0 | 271✓ d=0, 271✓ d=0, 271✓ d=0 | 271✓ d=0, 271✓ d=0, 271✓ d=0 | yes |
| M33 | M | T2 | 248 | (570,315)✓ d=0, (600,340)✓ d=0, (585,318)✓ d=0 | 2433✓ d=0, 2433✓ d=0, 2433✓ d=0 | 2433✓ d=0, 2433✓ d=0, 2433✓ d=0 | yes |
| N01 | N | T2 | 164 | (414,301)✗ d=5, (414,301)✗ d=5, (414,301)✗ d=5 | 190✓ d=0, 190✓ d=0, 190✓ d=0 | 190✓ d=0, 190✓ d=0, 190✓ d=0 | yes |
| N18 | N | T2 | 253 | (279,423)✓ d=0, (280,423)✓ d=0, (280,423)✓ d=0 | 536✓ d=0, 536✓ d=0, 536✓ d=0 | 536✓ d=0, 536✓ d=0, 536✓ d=0 | yes |
| N21 | N | T2 | 107 | (425,335)✓ d=0, (385,305)✓ d=0, (390,313)✓ d=0 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | yes |
| N26 | N | T2 | 112 | (375,298)✓ d=0, (370,298)✓ d=0, (375,298)✓ d=0 | 313✓ d=0, 313✓ d=0, 313✓ d=0 | 333✓ d=0, 333✓ d=0, 333✓ d=0 | yes |
| N34 | N | T2 | 117 | (520,630)✗ d=19, (520,640)✗ d=18, (530,630)✗ d=16 | 901✓ d=0, 901✓ d=0, 901✓ d=0 | 901✓ d=0, 901✓ d=0, 901✓ d=0 | yes |
| N40 | N | T2 | 90 | (447,350)✗ d=6, (447,349)✗ d=6, (447,349)✗ d=6 | 194✓ d=0, 194✓ d=0, 194✓ d=0 | 194✓ d=0, 201✓ d=0, 201✓ d=0 | yes |
| V37 | V | T2 | 65 | (487,459)✓ d=0, (488,461)✓ d=0, (487,462)✓ d=0 | 420✓ d=0, 420✓ d=0, 420✓ d=0 | 420✓ d=0, 420✓ d=0, 420✓ d=0 | yes |
| V46 | V | T2 | 274 | (565,272)✓ d=0, (540,272)✓ d=0, (530,265)✓ d=0 | 4037✓ d=0, 4037✓ d=0, 4037✓ d=0 | 4037✓ d=0, 4037✓ d=0, 4037✓ d=0 | yes |
| V54 | V | T2 | 27 | (333,195)✗ d=1, (330,220)✗ d=1, (330,215)✗ d=1 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | yes |
| V55 | V | T2 | 37 | (251,378)✓ d=0, (251,379)✓ d=0, (251,378)✓ d=0 | 69✓ d=0, 69✓ d=0, 69✓ d=0 | 69✓ d=0, 69✓ d=0, 69✓ d=0 | yes |
| V57 | V | T2 | 222 | (300,207)✓ d=0, (300,205)✓ d=0, (300,205)✓ d=0 | 223✓ d=0, 223✓ d=0, 223✓ d=0 | 223✓ d=0, 223✓ d=0, 223✓ d=0 | yes |
| V65 | V | T2 | 66 | (465,503)✓ d=0, (425,482)✓ d=0, (430,485)✓ d=0 | 222✓ d=0, 222✓ d=0, 222✓ d=0 | 222✓ d=0, 222✓ d=0, 222✓ d=0 | yes |

## Scoring rules
- S family: chosen mark's projected voxel inside GT on k*; `parse_fail` / `invalid_mark` / response errors count as misses and are reported separately.
- Majority vote per case over reps; ties broken by lowest mark id (S) / smallest (x, y) (R). `strict_majority_hit_rate` in score.json is restricted to cases where ≥2 reps agree.
- R: `panel_to_voxel(x, y)` inside GT on k*; a point outside the image panel = miss.
- distance: 0 if inside; else 2D EDT to the nearest GT voxel on k* (mm).
