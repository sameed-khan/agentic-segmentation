# Scores — phase1-02 (2026-10-18)

Cases scored: 30 (ok 30) | harness_sha256: `96903886bae040c2fd5559d42899aab6d5a28eb455eb38ce22e580744ad76a39` | model: anthropic/claude-sonnet-5-5

Dice/HD95/NSD of final_mask.nii.gz vs full GT; rules in eval/v2/score.py docstring.

| case | histotype | dice | HD95 mm | NSD@2mm | best | best dice | final==best | first click in GT | dist mm | tag | flags | oneclick | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F14 | F | 0 | 88.1913 | 0 | v1 | 0 | ✓ | no | 44.057 | localization | wrong_structure | 0.751 | -0.751 |
| F15 | F | 0 | 27.0185 | 0 | v1 | 0 | ✓ | no | 10.247 | localization | wrong_structure | 0.197 | -0.197 |
| F32 | F | 0.4147 | 97 | 0.1946 | v2 | 0.4147 | ✓ | yes | 0 | delineation | — | 0.407 | +0.008 |
| F44 | F | 0.7177 | 28.4429 | 0.2917 | v1 | 0.7177 | ✓ | yes | 0 | none | — | 0.682 | +0.036 |
| F49 | F | 0.9139 | 7.2111 | 0.5721 | v2 | 0.9139 | ✓ | yes | 0 | none | — | 0.866 | +0.048 |
| F50 | F | 0.8819 | 41.0506 | 0.5236 | v1 | 0.8819 | ✓ | yes | 0 | none | — | 0.448 | +0.434 |
| L19 | L | 0.8086 | 24.4949 | 0.4971 | v1 | 0.8086 | ✓ | yes | 0 | none | — | 0.825 | -0.016 |
| L20 | L | 0.9407 | 4.5826 | 0.7977 | v1 | 0.9407 | ✓ | yes | 0 | none | — | 0.941 | -0.000 |
| L33 | L | 0.0002 | 82.1097 | 0.0038 | v1 | 0.0002 | ✓ | no | 37.947 | localization | wrong_structure | 0.597 | -0.597 |
| L35 | L | 0.4868 | 39.4557 | 0.1485 | v2 | 0.4868 | ✓ | yes | 0 | delineation | — | 0.632 | -0.146 |
| L38 | L | 0.967 | 2 | 0.968 | v1 | 0.967 | ✓ | yes | 0 | none | — | 0.965 | +0.002 |
| L43 | L | 0 | 119.0714 | 0 | v1 | 0 | ✓ | no | 77.647 | localization | wrong_structure | 0.361 | -0.361 |
| M01 | M | 0.907 | 3 | 0.8447 | v1 | 0.907 | ✓ | yes | 0 | none | — | 0.899 | +0.008 |
| M07 | M | 0.9467 | 2 | 0.9632 | v1 | 0.9467 | ✓ | yes | 0 | none | — | 0.946 | +0.001 |
| M08 | M | 0.7363 | 21.4709 | 0.4415 | v1 | 0.7363 | ✓ | yes | 0 | none | — | 0.727 | +0.009 |
| M09 | M | 0.9538 | 2 | 0.9745 | v1 | 0.9538 | ✓ | yes | 0 | none | — | 0.950 | +0.003 |
| M21 | M | 0.8304 | 15.6844 | 0.4715 | v1 | 0.8342 | ✗ | yes | 0 | none | — | 0.815 | +0.015 |
| M33 | M | 0.5303 | 67.1267 | 0.088 | v1 | 0.5355 | ✗ | yes | 0 | none | — | 0.650 | -0.120 |
| N01 | N | 0.4771 | 17.8745 | 0.4262 | v1 | 0.4771 | ✓ | yes | 0 | delineation | — | 0.484 | -0.007 |
| N18 | N | 0.8862 | 12.3693 | 0.721 | v2 | 0.8862 | ✓ | yes | 0 | none | — | 0.883 | +0.003 |
| N21 | N | 0.8713 | 3 | 0.8953 | v1 | 0.8713 | ✓ | yes | 0 | none | — | 0.837 | +0.034 |
| N26 | N | 0.9276 | 4.1231 | 0.8462 | v1 | 0.9276 | ✓ | yes | 0 | none | — | 0.932 | -0.005 |
| N34 | N | 0.4942 | 32.28 | 0.2204 | v1 | 0.4942 | ✓ | yes | 0 | delineation | — | 0.492 | +0.002 |
| N40 | N | 0 | 132.2104 | 0 | v1 | 0 | ✓ | no | 46.195 | localization | wrong_structure | 0.280 | -0.280 |
| V37 | V | 0.7646 | 14.7309 | 0.7409 | v1 | 0.7646 | ✓ | yes | 0 | none | — | 0.770 | -0.005 |
| V46 | V | 0.8603 | 12.0416 | 0.5792 | v4 | 0.8603 | ✓ | yes | 0 | none | — | 0.608 | +0.252 |
| V54 | V | 0.7068 | 10.7703 | 0.7545 | v1 | 0.7068 | ✓ | yes | 0 | none | — | 0.785 | -0.079 |
| V55 | V | 0.8431 | 2 | 0.9819 | v1 | 0.8431 | ✓ | yes | 0 | none | — | 0.844 | -0.001 |
| V57 | V | 0 | 74.9133 | 0 | v1 | 0 | ✓ | no | 67.298 | localization | wrong_structure | 0.717 | -0.717 |
| V65 | V | 0.4949 | 40.5216 | 0.1683 | v2 | 0.4949 | ✓ | yes | 0 | delineation | — | 0.613 | -0.118 |

## Per-histotype Dice (final mask, vs full GT)

| histotype | n | mean | median | published U-Net mean | Δ vs U-Net | oneclick mean | Δ vs oneclick |
|---|---|---|---|---|---|---|---|
| F | 6 | 0.488 | 0.566 | 0.507 | -0.019 | 0.559 | -0.071 |
| L | 6 | 0.534 | 0.648 | 0.744 | -0.210 | 0.720 | -0.186 |
| M | 6 | 0.817 | 0.869 | 0.823 | -0.006 | 0.831 | -0.014 |
| N | 6 | 0.609 | 0.683 | 0.748 | -0.139 | 0.651 | -0.042 |
| V | 6 | 0.612 | 0.736 | 0.584 | +0.028 | 0.723 | -0.111 |
| **overall** | 30 | 0.612 | 0.750 | 0.681 | -0.069 | 0.697 | -0.085 |

## Notes

- F14 (F): flags wrong_structure (dice=0, tag=localization).
- F15 (F): flags wrong_structure (dice=0, tag=localization).
- L33 (L): flags wrong_structure (dice=0.0002, tag=localization).
- L43 (L): flags wrong_structure (dice=0, tag=localization).
- N40 (N): flags wrong_structure (dice=0, tag=localization).
- V57 (V): flags wrong_structure (dice=0, tag=localization).

