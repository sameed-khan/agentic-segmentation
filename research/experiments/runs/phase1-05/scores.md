# Scores — phase1-05 (2026-10-18)

Cases scored: 30 (ok 30) | harness_sha256: `3c915449843a9088674884fab22e3535f68b0693cbfa6a9e096a5e2011520782` | model: anthropic/claude-sonnet-5-5

Dice/HD95/NSD of final_mask.nii.gz vs full GT; rules in eval/v2/score.py docstring.

| case | histotype | dice | HD95 mm | NSD@2mm | best | best dice | final==best | first click in GT | dist mm | tag | flags | oneclick | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F14 | F | 0 | 147.057 | 0 | v1 | 0 | ✓ | no | 72.863 | localization | wrong_structure | 0.751 | -0.751 |
| F15 | F | 0.002 | 49.8116 | 0.0391 | v1 | 0.002 | ✗ | no | 10.247 | localization | wrong_structure | 0.197 | -0.195 |
| F32 | F | 0.4204 | 91.3729 | 0.2036 | v2 | 0.4204 | ✓ | yes | 0 | delineation | — | 0.407 | +0.013 |
| F44 | F | 0.8848 | 18.6815 | 0.586 | v2 | 0.8848 | ✓ | yes | 0 | none | — | 0.682 | +0.203 |
| F49 | F | 0.9045 | 8 | 0.5529 | v1 | 0.9045 | ✓ | yes | 0 | none | — | 0.866 | +0.038 |
| F50 | F | 0.8892 | 42.9418 | 0.5547 | v1 | 0.8892 | ✓ | yes | 0 | none | — | 0.448 | +0.441 |
| L19 | L | 0.8377 | 20.8567 | 0.5578 | v1 | 0.8377 | ✓ | yes | 0 | none | — | 0.825 | +0.013 |
| L20 | L | 0.9405 | 4.5826 | 0.7961 | v1 | 0.9405 | ✓ | yes | 0 | none | — | 0.941 | -0.000 |
| L33 | L | 0.0001 | 83.3079 | 0.003 | v1 | 0.0001 | ✓ | no | 21 | localization | wrong_structure | 0.597 | -0.597 |
| L35 | L | 0.5289 | 31.1288 | 0.1109 | v1 | 0.5289 | ✗ | yes | 0 | none | — | 0.632 | -0.104 |
| L38 | L | 0.967 | 2 | 0.968 | v1 | 0.967 | ✓ | yes | 0 | none | — | 0.965 | +0.002 |
| L43 | L | 0 | 116.0172 | 0 | v1 | 0 | ✓ | no | 77.647 | localization | wrong_structure | 0.361 | -0.361 |
| M01 | M | 0.907 | 3 | 0.8447 | v1 | 0.907 | ✓ | yes | 0 | none | — | 0.899 | +0.008 |
| M07 | M | 0.9467 | 2 | 0.9632 | v1 | 0.9467 | ✓ | yes | 0 | none | — | 0.946 | +0.001 |
| M08 | M | 0.7363 | 21.4709 | 0.4415 | v1 | 0.7363 | ✓ | yes | 0 | none | — | 0.727 | +0.009 |
| M09 | M | 0.9538 | 2 | 0.9745 | v1 | 0.9538 | ✓ | yes | 0 | none | — | 0.950 | +0.003 |
| M21 | M | 0.8149 | 16.3401 | 0.4155 | v1 | 0.819 | ✗ | yes | 0 | none | — | 0.815 | +0.000 |
| M33 | M | 0.5306 | 61.1474 | 0.0914 | v1 | 0.5306 | ✗ | yes | 0 | none | — | 0.650 | -0.120 |
| N01 | N | 0.4725 | 17.8326 | 0.4154 | v1 | 0.4725 | ✓ | yes | 0 | delineation | — | 0.484 | -0.012 |
| N18 | N | 0.8755 | 13 | 0.6623 | v1 | 0.8867 | ✗ | yes | 0 | none | — | 0.883 | -0.007 |
| N21 | N | 0.8564 | 3.6056 | 0.8586 | v1 | 0.8564 | ✓ | yes | 0 | none | — | 0.837 | +0.020 |
| N26 | N | 0.9295 | 4 | 0.8551 | v1 | 0.9295 | ✓ | yes | 0 | none | — | 0.932 | -0.003 |
| N34 | N | 0.5861 | 31.3209 | 0.342 | v1 | 0.5861 | ✓ | yes | 0 | none | — | 0.492 | +0.094 |
| N40 | N | 0 | 117.1561 | 0 | v1 | 0 | ✓ | no | 61.408 | localization | wrong_structure | 0.280 | -0.280 |
| V37 | V | 0.7646 | 14.7309 | 0.7409 | v1 | 0.7646 | ✓ | yes | 0 | none | — | 0.770 | -0.005 |
| V46 | V | 0 | 118.0635 | 0 | v1 | 0 | ✓ | no | 43.37 | localization | wrong_structure | 0.608 | -0.608 |
| V54 | V | 0.7068 | 10.7703 | 0.7545 | v1 | 0.7068 | ✓ | yes | 0 | none | — | 0.785 | -0.079 |
| V55 | V | 0.8454 | 2 | 0.9892 | v1 | 0.8454 | ✓ | no | 1 | none | — | 0.844 | +0.001 |
| V57 | V | 0 | 74.9133 | 0 | v1 | 0 | ✓ | no | 67.298 | localization | wrong_structure | 0.717 | -0.717 |
| V65 | V | 0.6181 | 36.3456 | 0.258 | v1 | 0.6181 | ✓ | yes | 0 | none | — | 0.613 | +0.005 |

## Per-histotype Dice (final mask, vs full GT)

| histotype | n | mean | median | published U-Net mean | Δ vs U-Net | oneclick mean | Δ vs oneclick |
|---|---|---|---|---|---|---|---|
| F | 6 | 0.517 | 0.653 | 0.507 | +0.010 | 0.559 | -0.042 |
| L | 6 | 0.546 | 0.683 | 0.744 | -0.198 | 0.720 | -0.174 |
| M | 6 | 0.815 | 0.861 | 0.823 | -0.008 | 0.831 | -0.016 |
| N | 6 | 0.620 | 0.721 | 0.748 | -0.128 | 0.651 | -0.031 |
| V | 6 | 0.489 | 0.662 | 0.584 | -0.095 | 0.723 | -0.234 |
| **overall** | 30 | 0.597 | 0.750 | 0.681 | -0.084 | 0.697 | -0.100 |

## Notes

- F14 (F): flags wrong_structure (dice=0, tag=localization).
- F15 (F): flags wrong_structure (dice=0.002, tag=localization).
- L33 (L): flags wrong_structure (dice=0.0001, tag=localization).
- L43 (L): flags wrong_structure (dice=0, tag=localization).
- N40 (N): flags wrong_structure (dice=0, tag=localization).
- V46 (V): flags wrong_structure (dice=0, tag=localization).
- V57 (V): flags wrong_structure (dice=0, tag=localization).

