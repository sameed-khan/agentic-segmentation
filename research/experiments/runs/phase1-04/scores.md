# Scores — phase1-04 (2026-10-18)

Cases scored: 30 (ok 30) | harness_sha256: `b9c4afcaf3d7505bdd602ae3a5d6d8ee08d0cd6c43f20eb18d6ee56cb14eba47` | model: anthropic/claude-sonnet-5-5

Dice/HD95/NSD of final_mask.nii.gz vs full GT; rules in eval/v2/score.py docstring.

| case | histotype | dice | HD95 mm | NSD@2mm | best | best dice | final==best | first click in GT | dist mm | tag | flags | oneclick | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F14 | F | 0 | 146.9415 | 0 | v1 | 0 | ✓ | no | 79.177 | localization | wrong_structure | 0.751 | -0.751 |
| F15 | F | 0 | 27.0185 | 0 | v1 | 0 | ✓ | no | 10.247 | localization | wrong_structure | 0.197 | -0.197 |
| F32 | F | 0.426 | 98.0255 | 0.1977 | v2 | 0.4261 | ✗ | yes | 0 | delineation | — | 0.407 | +0.019 |
| F44 | F | 0.725 | 28.7402 | 0.3026 | v2 | 0.725 | ✓ | yes | 0 | none | — | 0.682 | +0.043 |
| F49 | F | 0.8572 | 11.7898 | 0.3866 | v1 | 0.8572 | ✓ | yes | 0 | none | — | 0.866 | -0.009 |
| F50 | F | 0.8721 | 36.2491 | 0.484 | v1 | 0.8721 | ✓ | yes | 0 | none | — | 0.448 | +0.424 |
| L19 | L | 0.8387 | 20.8327 | 0.5573 | v1 | 0.8387 | ✓ | yes | 0 | none | — | 0.825 | +0.014 |
| L20 | L | 0.9431 | 4.3589 | 0.811 | v1 | 0.9431 | ✓ | yes | 0 | none | — | 0.941 | +0.002 |
| L33 | L | 0.0002 | 82.1584 | 0.0037 | v1 | 0.0002 | ✓ | no | 22.627 | localization | wrong_structure | 0.597 | -0.597 |
| L35 | L | 0.5289 | 31.1288 | 0.1109 | v1 | 0.5289 | ✗ | yes | 0 | none | — | 0.632 | -0.104 |
| L38 | L | 0.967 | 2 | 0.968 | v1 | 0.967 | ✓ | yes | 0 | none | — | 0.965 | +0.002 |
| L43 | L | 0 | 115.6315 | 0 | v1 | 0 | ✓ | no | 77.647 | localization | wrong_structure | 0.361 | -0.361 |
| M01 | M | 0.907 | 3 | 0.8447 | v1 | 0.907 | ✓ | yes | 0 | none | — | 0.899 | +0.008 |
| M07 | M | 0.9467 | 2 | 0.9632 | v1 | 0.9467 | ✓ | yes | 0 | none | — | 0.946 | +0.001 |
| M08 | M | 0.7363 | 21.4709 | 0.4415 | v1 | 0.7363 | ✓ | yes | 0 | none | — | 0.727 | +0.009 |
| M09 | M | 0.9538 | 2 | 0.9745 | v1 | 0.9538 | ✓ | yes | 0 | none | — | 0.950 | +0.003 |
| M21 | M | 0.8193 | 15.3623 | 0.4443 | v1 | 0.8193 | ✓ | yes | 0 | none | — | 0.815 | +0.004 |
| M33 | M | 0.528 | 65.681 | 0.0945 | v2 | 0.528 | ✓ | yes | 0 | none | — | 0.650 | -0.122 |
| N01 | N | 0.4771 | 17.8745 | 0.4262 | v1 | 0.4771 | ✓ | yes | 0 | delineation | — | 0.484 | -0.007 |
| N18 | N | 0.8524 | 15.6525 | 0.6404 | v1 | 0.8524 | ✓ | yes | 0 | none | — | 0.883 | -0.030 |
| N21 | N | 0.8531 | 3.7417 | 0.8398 | v1 | 0.8713 | ✗ | yes | 0 | none | — | 0.837 | +0.016 |
| N26 | N | 0.9276 | 4.1231 | 0.8462 | v1 | 0.9276 | ✓ | yes | 0 | none | — | 0.932 | -0.005 |
| N34 | N | 0.4942 | 32.28 | 0.2204 | v1 | 0.4942 | ✓ | yes | 0 | delineation | — | 0.492 | +0.002 |
| N40 | N | 0 | 136.2549 | 0 | v1 | 0 | ✓ | no | 47.106 | localization | wrong_structure | 0.280 | -0.280 |
| V37 | V | 0.7646 | 14.7309 | 0.7409 | v1 | 0.7646 | ✓ | yes | 0 | none | — | 0.770 | -0.005 |
| V46 | V | 0.8455 | 13.0767 | 0.5701 | v4 | 0.8455 | ✓ | yes | 0 | none | — | 0.608 | +0.237 |
| V54 | V | 0.7068 | 10.7703 | 0.7545 | v1 | 0.7068 | ✓ | yes | 0 | none | — | 0.785 | -0.079 |
| V55 | V | 0.8454 | 2 | 0.9892 | v1 | 0.8454 | ✓ | no | 1 | none | — | 0.844 | +0.001 |
| V57 | V | 0 | 63.3257 | 0 | v1 | 0 | ✓ | no | 57.184 | localization | wrong_structure | 0.717 | -0.717 |
| V65 | V | 0.5652 | 39.6106 | 0.1779 | v1 | 0.5788 | ✗ | yes | 0 | none | — | 0.613 | -0.048 |

## Per-histotype Dice (final mask, vs full GT)

| histotype | n | mean | median | published U-Net mean | Δ vs U-Net | oneclick mean | Δ vs oneclick |
|---|---|---|---|---|---|---|---|
| F | 6 | 0.480 | 0.576 | 0.507 | -0.027 | 0.559 | -0.079 |
| L | 6 | 0.546 | 0.684 | 0.744 | -0.198 | 0.720 | -0.174 |
| M | 6 | 0.815 | 0.863 | 0.823 | -0.008 | 0.831 | -0.016 |
| N | 6 | 0.601 | 0.673 | 0.748 | -0.147 | 0.651 | -0.050 |
| V | 6 | 0.621 | 0.736 | 0.584 | +0.037 | 0.723 | -0.102 |
| **overall** | 30 | 0.613 | 0.750 | 0.681 | -0.068 | 0.697 | -0.084 |

## Notes

- F14 (F): flags wrong_structure (dice=0, tag=localization).
- F15 (F): flags wrong_structure (dice=0, tag=localization).
- L33 (L): flags wrong_structure (dice=0.0002, tag=localization).
- L43 (L): flags wrong_structure (dice=0, tag=localization).
- N40 (N): flags wrong_structure (dice=0, tag=localization).
- V57 (V): flags wrong_structure (dice=0, tag=localization).

