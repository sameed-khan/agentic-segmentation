# Scores — phase1-06-high (2026-10-18)

Cases scored: 30 (ok 30) | harness_sha256: `61aae784cd7a733fdbea01df46a666b9a3cb0ce14b7426e4f3689872e4f4e931` | model: anthropic/claude-sonnet-5-5

Dice/HD95/NSD of final_mask.nii.gz vs full GT; rules in eval/v2/score.py docstring.

| case | histotype | dice | HD95 mm | NSD@2mm | best | best dice | final==best | first click in GT | dist mm | tag | flags | oneclick | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F14 | F | 0 | 106.1037 | 0 | v2 | 0.0001 | ✗ | no | 45.891 | localization | wrong_structure | 0.751 | -0.751 |
| F15 | F | 0 | 84.8664 | 0 | v1 | 0 | ✓ | no | 58.796 | localization | wrong_structure | 0.197 | -0.197 |
| F32 | F | 0.4482 | 78.6066 | 0.2137 | v2 | 0.4482 | ✓ | yes | 0 | delineation | — | 0.407 | +0.041 |
| F44 | F | 0.8859 | 18.7736 | 0.5899 | v1 | 0.8859 | ✓ | yes | 0 | none | — | 0.682 | +0.204 |
| F49 | F | 0.8526 | 12.8062 | 0.3765 | v1 | 0.8526 | ✓ | yes | 0 | none | — | 0.866 | -0.014 |
| F50 | F | 0.8771 | 39.2556 | 0.5044 | v1 | 0.8771 | ✓ | yes | 0 | none | — | 0.448 | +0.429 |
| L19 | L | 0.8695 | 17.2337 | 0.6284 | v9 | 0.8695 | ✓ | yes | 0 | none | — | 0.825 | +0.045 |
| L20 | L | 0.9453 | 4.1231 | 0.8267 | v1 | 0.9453 | ✓ | yes | 0 | none | — | 0.941 | +0.005 |
| L33 | L | 0.0002 | 82.2314 | 0.004 | v1 | 0.0002 | ✓ | no | 40.902 | localization | wrong_structure | 0.597 | -0.597 |
| L35 | L | 0.8996 | 12.7279 | 0.6127 | v3 | 0.8996 | ✓ | yes | 0 | none | — | 0.632 | +0.267 |
| L38 | L | 0.967 | 2 | 0.968 | v1 | 0.967 | ✓ | yes | 0 | none | — | 0.965 | +0.002 |
| L43 | L | 0.726 | 25.8592 | 0.2642 | v3 | 0.726 | ✓ | yes | 0 | none | — | 0.361 | +0.365 |
| M01 | M | 0.8918 | 4.3589 | 0.7896 | v1 | 0.8918 | ✓ | yes | 0 | none | — | 0.899 | -0.007 |
| M07 | M | 0.9467 | 2 | 0.9632 | v1 | 0.9467 | ✓ | yes | 0 | none | — | 0.946 | +0.001 |
| M08 | M | 0.7363 | 21.4709 | 0.4415 | v1 | 0.7363 | ✓ | yes | 0 | none | — | 0.727 | +0.009 |
| M09 | M | 0.9538 | 2 | 0.9745 | v1 | 0.9538 | ✓ | yes | 0 | none | — | 0.950 | +0.003 |
| M21 | M | 0.8351 | 15.1658 | 0.5026 | v2 | 0.8351 | ✓ | yes | 0 | none | — | 0.815 | +0.020 |
| M33 | M | 0.5185 | 73.3348 | 0.0835 | v1 | 0.5344 | ✗ | yes | 0 | none | — | 0.650 | -0.132 |
| N01 | N | 0 | 142.6499 | 0 | v1 | 0 | ✓ | no | 118.461 | localization | wrong_structure | 0.484 | -0.484 |
| N18 | N | 0.8891 | 13.3417 | 0.7491 | v2 | 0.8891 | ✓ | yes | 0 | none | — | 0.883 | +0.006 |
| N21 | N | 0.8531 | 3.7417 | 0.8398 | v1 | 0.8713 | ✗ | yes | 0 | none | — | 0.837 | +0.016 |
| N26 | N | 0.9319 | 4 | 0.8657 | v1 | 0.9319 | ✓ | yes | 0 | none | — | 0.932 | -0.000 |
| N34 | N | 0.4419 | 39.2046 | 0.1625 | v1 | 0.4419 | ✓ | yes | 0 | delineation | — | 0.492 | -0.050 |
| N40 | N | 0 | 150.0582 | 0 | v1 | 0 | ✓ | no | 62.209 | localization | wrong_structure | 0.280 | -0.280 |
| V37 | V | 0.7697 | 14.1986 | 0.748 | v1 | 0.7697 | ✓ | yes | 0 | none | — | 0.770 | -0.000 |
| V46 | V | 0.5983 | 32.0624 | 0.3475 | v1 | 0.5983 | ✓ | yes | 0 | none | — | 0.608 | -0.010 |
| V54 | V | 0.7761 | 10.6771 | 0.8453 | v1 | 0.7761 | ✓ | yes | 0 | none | — | 0.785 | -0.009 |
| V55 | V | 0.8415 | 1.7321 | 0.9942 | v1 | 0.8415 | ✓ | yes | 0 | none | — | 0.844 | -0.003 |
| V57 | V | 0.6978 | 5.6569 | 0.8136 | v1 | 0.6978 | ✓ | yes | 0 | none | — | 0.717 | -0.019 |
| V65 | V | 0.6174 | 38.4357 | 0.2544 | v1 | 0.6233 | ✗ | yes | 0 | none | — | 0.613 | +0.005 |

## Per-histotype Dice (final mask, vs full GT)

| histotype | n | mean | median | published U-Net mean | Δ vs U-Net | oneclick mean | Δ vs oneclick |
|---|---|---|---|---|---|---|---|
| F | 6 | 0.511 | 0.650 | 0.507 | +0.004 | 0.559 | -0.048 |
| L | 6 | 0.735 | 0.885 | 0.744 | -0.009 | 0.720 | +0.015 |
| M | 6 | 0.814 | 0.863 | 0.823 | -0.009 | 0.831 | -0.017 |
| N | 6 | 0.519 | 0.648 | 0.748 | -0.229 | 0.651 | -0.132 |
| V | 6 | 0.717 | 0.734 | 0.584 | +0.133 | 0.723 | -0.006 |
| **overall** | 30 | 0.659 | 0.806 | 0.681 | -0.022 | 0.697 | -0.038 |

## Notes

- F14 (F): flags wrong_structure (dice=0, tag=localization).
- F15 (F): flags wrong_structure (dice=0, tag=localization).
- L33 (L): flags wrong_structure (dice=0.0002, tag=localization).
- N01 (N): flags wrong_structure (dice=0, tag=localization).
- N40 (N): flags wrong_structure (dice=0, tag=localization).

