# Scores — phase1-01 (2026-10-18)

Cases scored: 30 (ok 30) | harness_sha256: `5cc4409b96847386fb828d684e642c797660c4c93c67d8b5aadda38f8788abd1` | model: anthropic/claude-sonnet-5-5

Dice/HD95/NSD of final_mask.nii.gz vs full GT; rules in eval/v2/score.py docstring.

| case | histotype | dice | HD95 mm | NSD@2mm | best | best dice | final==best | first click in GT | dist mm | tag | flags | oneclick | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F14 | F | 0.6451 | 16.1555 | 0.3925 | v1 | 0.6451 | ✓ | yes | 0 | none | — | 0.751 | -0.106 |
| F15 | F | 0 | 27.0185 | 0 | v1 | 0 | ✓ | no | 10.247 | localization | wrong_structure | 0.197 | -0.197 |
| F32 | F | 0.423 | 83.0602 | 0.2038 | v2 | 0.423 | ✓ | no | 3 | localization | — | 0.407 | +0.016 |
| F44 | F | 0.7248 | 25.5734 | 0.2735 | v2 | 0.7248 | ✓ | yes | 0 | none | — | 0.682 | +0.043 |
| F49 | F | 0.9045 | 7.874 | 0.5536 | v1 | 0.9045 | ✓ | yes | 0 | none | — | 0.866 | +0.038 |
| F50 | F | 0.8892 | 42.9418 | 0.5547 | v1 | 0.8892 | ✓ | yes | 0 | none | — | 0.448 | +0.441 |
| L19 | L | 0.8354 | 20.9284 | 0.5447 | v1 | 0.8354 | ✓ | yes | 0 | none | — | 0.825 | +0.011 |
| L20 | L | 0.9404 | 4.6904 | 0.7994 | v1 | 0.9404 | ✓ | yes | 0 | none | — | 0.941 | -0.000 |
| L33 | L | 0.0003 | 81.2773 | 0.0041 | v1 | 0.0003 | ✓ | no | 38.38 | localization | wrong_structure | 0.597 | -0.597 |
| L35 | L | 0.9228 | 8.3066 | 0.6468 | v4 | 0.9228 | ✓ | yes | 0 | none | — | 0.632 | +0.290 |
| L38 | L | 0.967 | 2 | 0.968 | v1 | 0.967 | ✓ | yes | 0 | none | — | 0.965 | +0.002 |
| L43 | L | 0 | 118.6971 | 0 | v1 | 0 | ✓ | no | 77.647 | localization | wrong_structure | 0.361 | -0.361 |
| M01 | M | 0.907 | 3 | 0.8447 | v1 | 0.907 | ✓ | yes | 0 | none | — | 0.899 | +0.008 |
| M07 | M | 0.9467 | 2 | 0.9632 | v1 | 0.9467 | ✓ | yes | 0 | none | — | 0.946 | +0.001 |
| M08 | M | 0.7363 | 21.4709 | 0.4415 | v1 | 0.7363 | ✓ | yes | 0 | none | — | 0.727 | +0.009 |
| M09 | M | 0.9538 | 2 | 0.9745 | v1 | 0.9538 | ✓ | yes | 0 | none | — | 0.950 | +0.003 |
| M21 | M | 0.8344 | 15.5242 | 0.5001 | v1 | 0.8344 | ✓ | yes | 0 | none | — | 0.815 | +0.020 |
| M33 | M | 0.473 | 72.0347 | 0.0799 | v1 | 0.473 | ✓ | yes | 0 | delineation | — | 0.650 | -0.177 |
| N01 | N | 0.4811 | 17.8326 | 0.4322 | v1 | 0.4811 | ✓ | yes | 0 | delineation | — | 0.484 | -0.003 |
| N18 | N | 0.888 | 11.3137 | 0.7443 | v1 | 0.888 | ✓ | yes | 0 | none | — | 0.883 | +0.005 |
| N21 | N | 0.8718 | 3 | 0.8944 | v2 | 0.8718 | ✓ | yes | 0 | none | — | 0.837 | +0.035 |
| N26 | N | 0.9228 | 4.4721 | 0.8187 | v1 | 0.9228 | ✓ | yes | 0 | none | — | 0.932 | -0.010 |
| N34 | N | 0.4419 | 39.2046 | 0.1625 | v1 | 0.4419 | ✓ | yes | 0 | delineation | — | 0.492 | -0.050 |
| N40 | N | 0 | 117.1561 | 0 | v1 | 0 | ✓ | no | 61.408 | localization | wrong_structure | 0.280 | -0.280 |
| V37 | V | 0.7646 | 14.7309 | 0.7409 | v1 | 0.7646 | ✓ | yes | 0 | none | — | 0.770 | -0.005 |
| V46 | V | 0 | 120.6897 | 0 | v1 | 0 | ✓ | no | 68.264 | localization | wrong_structure | 0.608 | -0.608 |
| V54 | V | 0.7068 | 10.7703 | 0.7545 | v1 | 0.7068 | ✓ | yes | 0 | none | — | 0.785 | -0.079 |
| V55 | V | 0.8431 | 2 | 0.9819 | v1 | 0.8431 | ✓ | yes | 0 | none | — | 0.844 | -0.001 |
| V57 | V | 0.7145 | 5.3852 | 0.8277 | v1 | 0.7145 | ✓ | yes | 0 | none | — | 0.717 | -0.003 |
| V65 | V | 0.434 | 43.0337 | 0.0997 | v1 | 0.434 | ✓ | yes | 0 | delineation | — | 0.613 | -0.179 |

## Per-histotype Dice (final mask, vs full GT)

| histotype | n | mean | median | published U-Net mean | Δ vs U-Net | oneclick mean | Δ vs oneclick |
|---|---|---|---|---|---|---|---|
| F | 6 | 0.598 | 0.685 | 0.507 | +0.091 | 0.559 | +0.039 |
| L | 6 | 0.611 | 0.879 | 0.744 | -0.133 | 0.720 | -0.109 |
| M | 6 | 0.809 | 0.871 | 0.823 | -0.014 | 0.831 | -0.022 |
| N | 6 | 0.601 | 0.676 | 0.748 | -0.147 | 0.651 | -0.050 |
| V | 6 | 0.577 | 0.711 | 0.584 | -0.007 | 0.723 | -0.146 |
| **overall** | 30 | 0.639 | 0.750 | 0.681 | -0.042 | 0.697 | -0.058 |

## Notes

- F15 (F): flags wrong_structure (dice=0, tag=localization).
- L33 (L): flags wrong_structure (dice=0.0003, tag=localization).
- L43 (L): flags wrong_structure (dice=0, tag=localization).
- N40 (N): flags wrong_structure (dice=0, tag=localization).
- V46 (V): flags wrong_structure (dice=0, tag=localization).

