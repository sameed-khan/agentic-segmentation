# Scores — integ-02 (2026-10-18)

Cases scored: 3 (ok 3) | harness_sha256: `c4a360f310d572ab201200e2a09da13022e8fa731e6cecda97eb04f6732955a9` | model: anthropic/claude-sonnet-5-5

Dice/HD95/NSD of final_mask.nii.gz vs full GT; rules in eval/v2/score.py docstring.

| case | histotype | dice | HD95 mm | NSD@2mm | best | best dice | final==best | first click in GT | dist mm | tag | flags | oneclick | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F14 | F | 0 | 91.7502 | 0 | v1 | 0 | ✓ | no | 39.749 | localization | wrong_structure | 0.751 | -0.751 |
| M33 | M | 0.1562 | 148.9597 | 0.0275 | v1 | 0.1562 | ✓ | yes | 0 | delineation | — | 0.650 | -0.494 |
| V55 | V | 0.8454 | 2 | 0.9892 | v1 | 0.8454 | ✓ | no | 1 | localization | — | 0.844 | +0.001 |

## Per-histotype Dice (final mask, vs full GT)

| histotype | n | mean | median | published U-Net mean | Δ vs U-Net | oneclick mean | Δ vs oneclick |
|---|---|---|---|---|---|---|---|
| F | 1 | 0.000 | 0.000 | 0.507 | -0.507 | 0.751 | -0.751 |
| M | 1 | 0.156 | 0.156 | 0.823 | -0.667 | 0.650 | -0.494 |
| V | 1 | 0.845 | 0.845 | 0.584 | +0.261 | 0.844 | +0.001 |
| **overall** | 3 | 0.334 | 0.156 | 0.681 | -0.347 | 0.749 | -0.415 |

## Notes

- F14 (F): flags wrong_structure (dice=0, tag=localization).

