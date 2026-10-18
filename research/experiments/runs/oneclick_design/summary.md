# One-click nnInteractive baseline — MSTT design cases

Cases: 30  |  succeeded: 30  |  failed/skipped: 0

One positive point at the GT largest-component centroid (snapped inside the component when the rounded centroid lands outside), no LLM. Dice is computed against the full GT mask.

| histotype | n | Dice (all) mean | Dice (all) median | Dice (LCC) mean | Dice (LCC) median | published U-Net mean (ref) |
|---|---|---|---|---|---|---|
| F | 6 | 0.559 | 0.565 | 0.533 | 0.565 | 0.507 |
| L | 6 | 0.720 | 0.729 | 0.720 | 0.729 | 0.744 |
| M | 6 | 0.831 | 0.857 | 0.831 | 0.857 | 0.823 |
| N | 6 | 0.651 | 0.665 | 0.655 | 0.665 | 0.748 |
| V | 6 | 0.723 | 0.743 | 0.736 | 0.743 | 0.584 |
| **overall** | 30 | 0.697 | 0.739 | 0.695 | 0.739 | 0.681 |

Mean timings over 30 cases: set_image 0.17s, prompt 0.77s.

## Notes

- The published U-Net means are for the full MSTT-199 dataset (n=199) and are **not directly comparable**: different case set, and here the prompt uses an oracle GT seed. Treat them as rough context only.

## Notable cases

- F15 (F): predicted volume 5.19 mL is >2x GT 2.167 mL (dice_all=0.1968, pred_components=1).
- F32 (F): predicted volume 841.153 mL is >2x GT 219.587 mL (dice_all=0.407, pred_components=5).
- N40 (N): predicted volume 2.526 mL is <0.2x GT 13.571 mL (dice_all=0.2797).

