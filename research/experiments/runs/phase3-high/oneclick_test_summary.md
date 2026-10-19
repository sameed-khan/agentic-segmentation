# One-click nnInteractive baseline — MSTT test cases

Cases: 169  |  succeeded: 169  |  failed/skipped: 0

One positive point at the GT largest-component centroid (snapped inside the component when the rounded centroid lands outside), no LLM. Dice is computed against the full GT mask.

| histotype | n | Dice (all) mean | Dice (all) median | Dice (LCC) mean | Dice (LCC) median | published U-Net mean (ref) |
|---|---|---|---|---|---|---|
| F | 34 | 0.625 | 0.735 | 0.634 | 0.735 | 0.507 |
| L | 34 | 0.626 | 0.776 | 0.626 | 0.776 | 0.744 |
| M | 34 | 0.811 | 0.876 | 0.811 | 0.876 | 0.823 |
| N | 34 | 0.781 | 0.863 | 0.783 | 0.863 | 0.748 |
| V | 33 | 0.733 | 0.791 | 0.746 | 0.823 | 0.584 |
| **overall** | 169 | 0.715 | 0.817 | 0.720 | 0.820 | 0.681 |

Mean timings over 169 cases: set_image 0.15s, prompt 0.44s.

## Notes

- The published U-Net means are for the full MSTT-199 dataset (n=199) and are **not directly comparable**: different case set, and here the prompt uses an oracle GT seed. Treat them as rough context only.

## Notable cases

- F18 (F): predicted volume 0.247 mL is <0.2x GT 542.45 mL (dice_all=0.0002).
- F21 (F): predicted volume 0.641 mL is <0.2x GT 1072.533 mL (dice_all=0.0012).
- F22 (F): predicted volume 0.122 mL is <0.2x GT 2.935 mL (dice_all=0.0798).
- F31 (F): predicted volume 55.719 mL is <0.2x GT 826.956 mL (dice_all=0.1263).
- F51 (F): predicted volume 10.994 mL is <0.2x GT 64.384 mL (dice_all=0.2901).
- L03 (L): predicted volume 1.283 mL is <0.2x GT 1373.749 mL (dice_all=0.0019).
- L09 (L): predicted volume 0.508 mL is <0.2x GT 389.837 mL (dice_all=0.0026).
- L12 (L): predicted volume 7.016 mL is <0.2x GT 578.223 mL (dice_all=0.0213).
- L17 (L): predicted volume 72.069 mL is <0.2x GT 1090.28 mL (dice_all=0.124).
- L23 (L): predicted volume 86.139 mL is <0.2x GT 1585.524 mL (dice_all=0.1031).
- L24 (L): predicted volume 0.936 mL is <0.2x GT 483.506 mL (dice_all=0.0039).
- L26 (L): predicted volume 44.266 mL is <0.2x GT 1683.683 mL (dice_all=0.0512).
- L32 (L): predicted volume 24.675 mL is <0.2x GT 731.431 mL (dice_all=0.0653).
- M25 (M): predicted volume 0.78 mL is <0.2x GT 509.439 mL (dice_all=0.003).
- N23 (N): predicted volume 0.681 mL is <0.2x GT 301.924 mL (dice_all=0.0045).
- N24 (N): predicted volume 263.487 mL is <0.2x GT 1747.261 mL (dice_all=0.2621).
- N28 (N): predicted volume 3.043 mL is <0.2x GT 774.227 mL (dice_all=0.0078).
- V24 (V): predicted volume 65.253 mL is <0.2x GT 347.149 mL (dice_all=0.1456).
- V36 (V): predicted volume 242.751 mL is <0.2x GT 1609.709 mL (dice_all=0.2621).
- V39 (V): predicted volume 4.528 mL is <0.2x GT 143.968 mL (dice_all=0.061).

