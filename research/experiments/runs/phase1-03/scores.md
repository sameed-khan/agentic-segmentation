# Scores — phase1-03 (2026-10-18)

Cases scored: 30 (ok 30) | harness_sha256: `d9cb6eae331b989296d808c215018578f840a4efeae8522f2fd20d276d171462` | model: anthropic/claude-sonnet-5-5

Dice/HD95/NSD of final_mask.nii.gz vs full GT; rules in eval/v2/score.py docstring.

| case | histotype | dice | HD95 mm | NSD@2mm | best | best dice | final==best | first click in GT | dist mm | tag | flags | oneclick | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F14 | F | 0 | 146.9522 | 0 | v1 | 0 | ✓ | no | 104.814 | localization | wrong_structure | 0.751 | -0.751 |
| F15 | F | 0 | 27.0185 | 0 | v1 | 0 | ✓ | no | 10.247 | localization | wrong_structure | 0.197 | -0.197 |
| F32 | F | 0.4253 | 85.5102 | 0.2127 | v2 | 0.4253 | ✓ | yes | 0 | delineation | — | 0.407 | +0.018 |
| F44 | F | 0.7179 | 28.4429 | 0.2927 | v2 | 0.7179 | ✓ | yes | 0 | none | — | 0.682 | +0.036 |
| F49 | F | 0.9036 | 8 | 0.55 | v1 | 0.9036 | ✓ | yes | 0 | none | — | 0.866 | +0.037 |
| F50 | F | 0.8819 | 41.0506 | 0.5236 | v1 | 0.8819 | ✓ | yes | 0 | none | — | 0.448 | +0.434 |
| L19 | L | 0.839 | 20.8327 | 0.5596 | v1 | 0.839 | ✓ | yes | 0 | none | — | 0.825 | +0.014 |
| L20 | L | 0.9431 | 4.3589 | 0.8136 | v1 | 0.9431 | ✓ | yes | 0 | none | — | 0.941 | +0.002 |
| L33 | L | 0.0003 | 81.111 | 0.0042 | v1 | 0.0003 | ✓ | no | 21 | localization | wrong_structure | 0.597 | -0.597 |
| L35 | L | 0.4761 | 70.8431 | 0.2793 | v2 | 0.4766 | ✗ | yes | 0 | delineation | — | 0.632 | -0.156 |
| L38 | L | 0.967 | 2 | 0.968 | v1 | 0.967 | ✓ | yes | 0 | none | — | 0.965 | +0.002 |
| L43 | L | 0 | 120.6234 | 0 | v1 | 0 | ✓ | no | 81.908 | localization | wrong_structure | 0.361 | -0.361 |
| M01 | M | 0.907 | 3 | 0.8447 | v1 | 0.907 | ✓ | yes | 0 | none | — | 0.899 | +0.008 |
| M07 | M | 0.9467 | 2 | 0.9632 | v1 | 0.9467 | ✓ | yes | 0 | none | — | 0.946 | +0.001 |
| M08 | M | 0.7363 | 21.4709 | 0.4415 | v1 | 0.7363 | ✓ | yes | 0 | none | — | 0.727 | +0.009 |
| M09 | M | 0.9538 | 2 | 0.9745 | v1 | 0.9538 | ✓ | yes | 0 | none | — | 0.950 | +0.003 |
| M21 | M | 0.8334 | 15.3623 | 0.4972 | v1 | 0.8334 | ✓ | yes | 0 | none | — | 0.815 | +0.019 |
| M33 | M | 0.1562 | 148.9597 | 0.0275 | v1 | 0.1562 | ✓ | yes | 0 | delineation | — | 0.650 | -0.494 |
| N01 | N | 0.4771 | 17.8745 | 0.4262 | v1 | 0.4771 | ✓ | yes | 0 | delineation | — | 0.484 | -0.007 |
| N18 | N | 0.892 | 11.9289 | 0.7328 | v1 | 0.892 | ✓ | yes | 0 | none | — | 0.883 | +0.009 |
| N21 | N | 0.8713 | 3 | 0.8953 | v1 | 0.8713 | ✓ | yes | 0 | none | — | 0.837 | +0.034 |
| N26 | N | 0.9276 | 4.1231 | 0.8462 | v1 | 0.9276 | ✓ | yes | 0 | none | — | 0.932 | -0.005 |
| N34 | N | 0 | 164.5266 | 0 | v1 | 0 | ✓ | no | 69.986 | localization | wrong_structure | 0.492 | -0.492 |
| N40 | N | 0 | 138.6162 | 0 | v1 | 0 | ✓ | no | 59.042 | localization | wrong_structure | 0.280 | -0.280 |
| V37 | V | 0.7646 | 14.7309 | 0.7409 | v1 | 0.7646 | ✓ | yes | 0 | none | — | 0.770 | -0.005 |
| V46 | V | 0.3647 | 39.1024 | 0.2295 | v1 | 0.3647 | ✓ | yes | 0 | delineation | — | 0.608 | -0.243 |
| V54 | V | 0.797 | 10.6301 | 0.8555 | v2 | 0.797 | ✓ | yes | 0 | none | — | 0.785 | +0.012 |
| V55 | V | 0.8454 | 2 | 0.9892 | v1 | 0.8454 | ✓ | no | 1 | none | — | 0.844 | +0.001 |
| V57 | V | 0 | 90.6959 | 0 | v1 | 0 | ✓ | no | 83.187 | localization | wrong_structure | 0.717 | -0.717 |
| V65 | V | 0.5714 | 40.025 | 0.1756 | v1 | 0.5801 | ✗ | yes | 0 | none | — | 0.613 | -0.041 |

## Per-histotype Dice (final mask, vs full GT)

| histotype | n | mean | median | published U-Net mean | Δ vs U-Net | oneclick mean | Δ vs oneclick |
|---|---|---|---|---|---|---|---|
| F | 6 | 0.488 | 0.572 | 0.507 | -0.019 | 0.559 | -0.071 |
| L | 6 | 0.538 | 0.658 | 0.744 | -0.206 | 0.720 | -0.182 |
| M | 6 | 0.756 | 0.870 | 0.823 | -0.067 | 0.831 | -0.075 |
| N | 6 | 0.528 | 0.674 | 0.748 | -0.220 | 0.651 | -0.123 |
| V | 6 | 0.557 | 0.668 | 0.584 | -0.027 | 0.723 | -0.166 |
| **overall** | 30 | 0.573 | 0.750 | 0.681 | -0.108 | 0.697 | -0.124 |

## Notes

- F14 (F): flags wrong_structure (dice=0, tag=localization).
- F15 (F): flags wrong_structure (dice=0, tag=localization).
- L33 (L): flags wrong_structure (dice=0.0003, tag=localization).
- L43 (L): flags wrong_structure (dice=0, tag=localization).
- N34 (N): flags wrong_structure (dice=0, tag=localization).
- N40 (N): flags wrong_structure (dice=0, tag=localization).
- V57 (V): flags wrong_structure (dice=0, tag=localization).

