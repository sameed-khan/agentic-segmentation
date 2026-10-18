# Mark coverage of GT lesions — design cases

Params: version=7, downsample_mm=2.0, region_downsample_mm=4.0, log_sigmas_mm=[2.0, 4.0, 8.0, 16.0], log_rel_thr=0.08, log_abs_thr=0.05, body_erode_mm=4.0, supervoxel_mm=20.0, supervoxel_compactness=0.5, supervoxel_max_iter=5, supervoxel_min_mm3=1000.0, fine_spacing_mm=5.0, fine_clear_coarse_mm=3.0, region_strength_mm=10.0, region_scale_clamp_mm=[8.0, 40.0], body_grid_max_dim=240
Cases: 30; thinning/label cap: visible() cap 50, in-plane thin coarse 10.0 mm / fine 6.0 mm

| metric | result |
|---|---|
| (a) 3D coverage (any mark voxel inside GT) | 30/30 = 1.00 |
| (b) in-slice, drawn coarse set, full axial slice at GT max | 30/30 = 1.00 |
| (c) in-slice, drawn fine set, full axial slice at GT max | 30/30 = 1.00 |
| (d) labels drawn (coarse): max 50, median 50 |
| (d) labels drawn (fine): max 50, median 50 |

Per histotype (passed/total):

- (a) F: 6/6; (b) coarse: 6/6; (c) fine: 6/6
- (a) L: 6/6; (b) coarse: 6/6; (c) fine: 6/6
- (a) M: 6/6; (b) coarse: 6/6; (c) fine: 6/6
- (a) N: 6/6; (b) coarse: 6/6; (c) fine: 6/6
- (a) V: 6/6; (b) coarse: 6/6; (c) fine: 6/6

Failing (b) — no drawn coarse mark inside GT on the largest axial slice:

- none

Small lesions (fine level on a crop around the lesion region):

- F15 (2.17 mL): crop fine hit True (dist 0.0 mm, 20 labels); coarse full-slice hit True
- V55 (1.95 mL): crop fine hit True (dist 0.0 mm, 13 labels); coarse full-slice hit True
- V57 (3.01 mL): crop fine hit True (dist 0.0 mm, 25 labels); coarse full-slice hit True
