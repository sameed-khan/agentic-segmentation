# Mark coverage of GT lesions — beamster design split (per lesion, profile=default)

Params: version=7, downsample_mm=2.0, region_downsample_mm=4.0, log_sigmas_mm=[2.0, 4.0, 8.0, 16.0], log_rel_thr=0.08, log_abs_thr=0.05, body_erode_mm=4.0, supervoxel_mm=20.0, supervoxel_compactness=0.5, supervoxel_max_iter=5, supervoxel_min_mm3=1000.0, fine_spacing_mm=5.0, fine_clear_coarse_mm=3.0, region_strength_mm=10.0, region_scale_clamp_mm=[8.0, 40.0], body_grid_max_dim=240
Units: 72 GT lesions on 30 cases; hit = a DRAWN mark (CONTRACTS projection) lands inside the lesion on its largest axial slice; full = full axial view, crop = fine marks drawn in the lesion crop (seg_view zoom).

| lesion size | n | 3D any mark | full coarse | full fine | crop fine |
|---|---|---|---|---|---|
| <0.5 mL | 29 | 16/29 = 0.55 | 3/29 = 0.10 | 3/29 = 0.10 | 16/29 = 0.55 |
| 0.5-2 mL | 17 | 17/17 = 1.00 | 7/17 = 0.41 | 7/17 = 0.41 | 17/17 = 1.00 |
| >2 mL | 26 | 26/26 = 1.00 | 22/26 = 0.85 | 22/26 = 0.85 | 26/26 = 1.00 |
| **all** | 72 | 59/72 | 32/72 | 32/72 | 59/72 |

Lesions with NO drawn mark even in the fine crop view:

- Dataset_A_036 L1 (0.04 mL, slice 248)
- Dataset_A_036 L6 (0.048 mL, slice 223)
- Dataset_A_120 L5 (0.386 mL, slice 315)
- Dataset_A_120 L6 (0.026 mL, slice 228)
- Dataset_A_121 L1 (0.254 mL, slice 240)
- Dataset_B_011 L1 (0.385 mL, slice 220)
- Dataset_B_011 L3 (0.087 mL, slice 165)
- Dataset_B_013 L1 (0.122 mL, slice 154)
- Dataset_B_016 L1 (0.108 mL, slice 195)
- Dataset_B_016 L2 (0.033 mL, slice 227)
- Dataset_B_016 L3 (0.029 mL, slice 209)
- Dataset_B_016 L5 (0.055 mL, slice 183)
- Dataset_B_016 L6 (0.031 mL, slice 168)
