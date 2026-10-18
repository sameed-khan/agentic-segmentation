# Phase 2 localization study — run `dev-02-high`

Model `anthropic/claude-sonnet-5-5`, thinking `high`, 3 reps × 30 design cases; scored 2026-10-18T23:32:18Z. pi SDK 1.2.0 (auth shim 3.4.3).

## Caveats
- **Dev cases**: the same 30 design cases the marks were tuned on → coverage is optimistic.
- **n = 6 per histotype** (one case ≈ 17 points) → the gate is a coarse go/no-go signal; report per-case results.
- Coverage counts drawn coarse marks inside GT on k* only (single-slice study; the full test-split study runs later).

## Headline

| condition | n | hit rate (mean over reps) | majority-vote hit rate (per case) | coverage | parse fail | invalid mark | outside panel | resp error | dist mean mm | dist median mm |
|---|---|---|---|---|---|---|---|---|---|---|
| R | 90 | 0.700 | 0.733 | – | 0 | 0 | 0 | 0 | 6.865 | 0.0 |
| S | 90 | 0.911 | 0.900 | 1.000 | 0 | 0 | 0 | 0 | 3.252 | 0.0 |
| S-ex | 90 | 0.900 | 0.900 | 1.000 | 0 | 0 | 0 | 0 | 4.344 | 0.0 |

Drawn-mark coverage on k*: 30/30 = 1.000.

## Gate (harness_design_v2.md §3, thresholds fixed before data)
- S hit rate (mean over reps): **0.911**; R: **0.700**; S − R: **+0.211**
- Gate (S ≥ 0.8 and S − R ≥ 0.1): **PASS** → proceed with S.

## Per histotype (hit rate, mean over reps)

| histotype | R | S | S-ex |
|---|---|---|---|
| F | 0.667 | 0.667 | 0.667 |
| L | 0.778 | 0.889 | 0.833 |
| M | 0.944 | 1.000 | 1.000 |
| N | 0.500 | 1.000 | 1.000 |
| V | 0.611 | 1.000 | 1.000 |

## Per-case table

| case | h | seq | k* | R | S | S-ex | covered |
|---|---|---|---|---|---|---|---|---|
| F14 | F | T2 | 90 | (350,270)✗ d=80, (350,265)✗ d=82, (350,268)✗ d=80 | 1161✗ d=62, 1161✗ d=62, 1161✗ d=62 | 1121✗ d=90, 1121✗ d=90, 1161✗ d=62 | yes |
| F15 | F | T2 | 90 | (395,387)✗ d=6, (395,385)✗ d=7, (393,385)✗ d=7 | 175✗ d=7, 175✗ d=7, 175✗ d=7 | 175✗ d=7, 175✗ d=7, 175✗ d=7 | yes |
| F32 | F | T2 | 198 | (535,410)✓ d=0, (545,430)✓ d=0, (530,420)✓ d=0 | 864✓ d=0, 864✓ d=0, 864✓ d=0 | 864✓ d=0, 864✓ d=0, 864✓ d=0 | yes |
| F44 | F | T2 | 75 | (420,370)✓ d=0, (420,365)✓ d=0, (425,365)✓ d=0 | 462✓ d=0, 462✓ d=0, 462✓ d=0 | 462✓ d=0, 462✓ d=0, 462✓ d=0 | yes |
| F49 | F | T2 | 132 | (515,480)✓ d=0, (515,485)✓ d=0, (520,485)✓ d=0 | 978✓ d=0, 978✓ d=0, 978✓ d=0 | 978✓ d=0, 978✓ d=0, 978✓ d=0 | yes |
| F50 | F | T2 | 52 | (370,580)✓ d=0, (370,585)✓ d=0, (370,585)✓ d=0 | 525✓ d=0, 467✓ d=0, 525✓ d=0 | 467✓ d=0, 467✓ d=0, 467✓ d=0 | yes |
| L19 | L | T1 | 259 | (360,570)✓ d=0, (370,570)✗ d=2, (355,575)✓ d=0 | 3707✓ d=0, 3707✓ d=0, 3707✓ d=0 | 3701✓ d=0, 3701✓ d=0, 3659✓ d=0 | yes |
| L20 | L | T1 | 144 | (480,340)✓ d=0, (480,340)✓ d=0, (480,340)✓ d=0 | 370✓ d=0, 370✓ d=0, 370✓ d=0 | 370✓ d=0, 370✓ d=0, 370✓ d=0 | yes |
| L33 | L | T2 | 119 | (390,370)✗ d=43, (390,370)✗ d=43, (390,370)✗ d=43 | 414✓ d=0, 409✗ d=42, 409✗ d=42 | 409✗ d=42, 409✗ d=42, 409✗ d=42 | yes |
| L35 | L | T1 | 147 | (450,540)✓ d=0, (480,550)✓ d=0, (450,540)✓ d=0 | 1383✓ d=0, 1383✓ d=0, 1383✓ d=0 | 1383✓ d=0, 1383✓ d=0, 1383✓ d=0 | yes |
| L38 | L | T1 | 97 | (385,425)✓ d=0, (385,425)✓ d=0, (388,420)✓ d=0 | 152✓ d=0, 152✓ d=0, 152✓ d=0 | 152✓ d=0, 152✓ d=0, 152✓ d=0 | yes |
| L43 | L | T1 | 196 | (480,290)✓ d=0, (480,290)✓ d=0, (480,290)✓ d=0 | 2954✓ d=0, 2954✓ d=0, 2954✓ d=0 | 2954✓ d=0, 2954✓ d=0, 2954✓ d=0 | yes |
| M01 | M | T2 | 133 | (253,492)✓ d=0, (253,492)✓ d=0, (253,493)✓ d=0 | 1674✓ d=0, 1674✓ d=0, 1674✓ d=0 | 1674✓ d=0, 1674✓ d=0, 1674✓ d=0 | yes |
| M07 | M | T2 | 80 | (475,575)✓ d=0, (475,574)✓ d=0, (475,574)✓ d=0 | 76✓ d=0, 76✓ d=0, 76✓ d=0 | 76✓ d=0, 76✓ d=0, 76✓ d=0 | yes |
| M08 | M | T2 | 70 | (380,402)✓ d=0, (380,402)✓ d=0, (380,400)✓ d=0 | 274✓ d=0, 274✓ d=0, 274✓ d=0 | 274✓ d=0, 274✓ d=0, 274✓ d=0 | yes |
| M09 | M | T2 | 108 | (202,402)✓ d=0, (201,401)✓ d=0, (200,401)✓ d=0 | 661✓ d=0, 661✓ d=0, 661✓ d=0 | 661✓ d=0, 661✓ d=0, 661✓ d=0 | yes |
| M21 | M | T2 | 304 | (300,360)✓ d=0, (310,360)✓ d=0, (310,345)✓ d=0 | 270✓ d=0, 270✓ d=0, 270✓ d=0 | 272✓ d=0, 259✓ d=0, 271✓ d=0 | yes |
| M33 | M | T2 | 248 | (610,335)✓ d=0, (620,340)✗ d=1, (610,335)✓ d=0 | 2433✓ d=0, 2433✓ d=0, 2433✓ d=0 | 2433✓ d=0, 2433✓ d=0, 2433✓ d=0 | yes |
| N01 | N | T2 | 164 | (455,440)✗ d=50, (457,442)✗ d=51, (458,442)✗ d=51 | 190✓ d=0, 190✓ d=0, 190✓ d=0 | 190✓ d=0, 190✓ d=0, 190✓ d=0 | yes |
| N18 | N | T2 | 253 | (318,444)✓ d=0, (318,446)✓ d=0, (318,445)✓ d=0 | 536✓ d=0, 536✓ d=0, 536✓ d=0 | 536✓ d=0, 536✓ d=0, 536✓ d=0 | yes |
| N21 | N | T2 | 107 | (425,330)✓ d=0, (425,335)✓ d=0, (425,328)✓ d=0 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | yes |
| N26 | N | T2 | 112 | (410,322)✓ d=0, (410,322)✓ d=0, (410,322)✓ d=0 | 313✓ d=0, 313✓ d=0, 313✓ d=0 | 302✓ d=0, 313✓ d=0, 285✓ d=0 | yes |
| N34 | N | T2 | 117 | (550,505)✗ d=4, (530,640)✗ d=15, (540,640)✗ d=12 | 901✓ d=0, 901✓ d=0, 901✓ d=0 | 901✓ d=0, 901✓ d=0, 901✓ d=0 | yes |
| N40 | N | T2 | 90 | (447,350)✗ d=6, (447,349)✗ d=6, (447,349)✗ d=6 | 194✓ d=0, 194✓ d=0, 194✓ d=0 | 194✓ d=0, 194✓ d=0, 194✓ d=0 | yes |
| V37 | V | T2 | 65 | (525,481)✗ d=4, (527,483)✗ d=4, (527,482)✗ d=4 | 420✓ d=0, 420✓ d=0, 420✓ d=0 | 420✓ d=0, 420✓ d=0, 420✓ d=0 | yes |
| V46 | V | T2 | 274 | (566,272)✓ d=0, (555,270)✓ d=0, (530,268)✓ d=0 | 4037✓ d=0, 4037✓ d=0, 4037✓ d=0 | 4037✓ d=0, 4037✓ d=0, 4037✓ d=0 | yes |
| V54 | V | T2 | 27 | (330,195)✓ d=0, (330,190)✓ d=0, (333,200)✗ d=1 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | 44✓ d=0, 44✓ d=0, 44✓ d=0 | yes |
| V55 | V | T2 | 37 | (290,399)✗ d=3, (290,399)✗ d=3, (291,399)✗ d=3 | 69✓ d=0, 69✓ d=0, 69✓ d=0 | 69✓ d=0, 69✓ d=0, 69✓ d=0 | yes |
| V57 | V | T2 | 222 | (297,208)✓ d=0, (298,207)✓ d=0, (298,205)✓ d=0 | 223✓ d=0, 223✓ d=0, 223✓ d=0 | 223✓ d=0, 223✓ d=0, 223✓ d=0 | yes |
| V65 | V | T2 | 66 | (470,505)✓ d=0, (470,503)✓ d=0, (470,505)✓ d=0 | 222✓ d=0, 222✓ d=0, 222✓ d=0 | 222✓ d=0, 222✓ d=0, 222✓ d=0 | yes |

## Scoring rules
- S family: chosen mark's projected voxel inside GT on k*; `parse_fail` / `invalid_mark` / response errors count as misses and are reported separately.
- Majority vote per case over reps; ties broken by lowest mark id (S) / smallest (x, y) (R). `strict_majority_hit_rate` in score.json is restricted to cases where ≥2 reps agree.
- R: `panel_to_voxel(x, y)` inside GT on k*; a point outside the image panel = miss.
- distance: 0 if inside; else 2D EDT to the nearest GT voxel on k* (mm).
