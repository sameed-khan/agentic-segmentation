# Candidate-correction benchmark

States: 240 requested, 33 skipped (final mask has no overlap with GT = localization failure), 207 evaluated for at least one method.
Oracle-accept = apply exactly the candidates a perfect chooser accepts (add: precision > .5, remove: > .5 outside GT). Probe candidates are applied as union/difference of the predicted regions (approximation of sequential clicking). Time excludes the body mask (once per case).

### All evaluated states (each method on its own state set)

| method | states | cand/state | FN-rec@1 | @3 | @8 | add prec | add prec>.5 | rem prec | start Dice | oracle-accept Dice | gain | accept-all Dice | accept-top1 Dice | time/state s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combo | 207 | 6.5 | 0.348 | 0.549 | 0.636 | 0.202 | 0.153 | 0.198 | 0.739 | 0.782 | +0.043 | 0.577 | 0.641 | 6.7 |
| ends | 207 | 3.3 | 0.035 | 0.064 | 0.074 | 0.243 | 0.221 | nan | 0.739 | 0.750 | +0.011 | 0.644 | 0.681 | 2.7 |
| intensity | 207 | 3.9 | 0.297 | 0.536 | 0.552 | 0.298 | 0.284 | nan | 0.739 | 0.786 | +0.047 | 0.641 | 0.681 | 1.6 |
| probe | 207 | 4.3 | 0.466 | 0.597 | 0.631 | 0.304 | 0.207 | 0.249 | 0.739 | 0.784 | +0.045 | 0.602 | 0.661 | 28.9 |
| supervoxel | 207 | 6.1 | 0.360 | 0.560 | 0.572 | 0.197 | 0.152 | 0.224 | 0.739 | 0.777 | +0.038 | 0.626 | 0.670 | 5.2 |

### Under-segmented states (start recall < 0.7)

| method | states | cand/state | FN-rec@1 | @3 | @8 | add prec | add prec>.5 | rem prec | start Dice | oracle-accept Dice | gain | accept-all Dice | accept-top1 Dice | time/state s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combo | 84 | 6.2 | 0.231 | 0.462 | 0.527 | 0.353 | 0.322 | 0.174 | 0.555 | 0.656 | +0.101 | 0.562 | 0.540 | 5.4 |
| ends | 84 | 2.9 | 0.072 | 0.110 | 0.124 | 0.416 | 0.364 | nan | 0.555 | 0.579 | +0.024 | 0.546 | 0.547 | 2.5 |
| intensity | 84 | 3.8 | 0.231 | 0.450 | 0.466 | 0.499 | 0.491 | nan | 0.555 | 0.663 | +0.108 | 0.622 | 0.598 | 1.2 |
| probe | 84 | 5.0 | 0.340 | 0.447 | 0.513 | 0.446 | 0.410 | 0.207 | 0.555 | 0.657 | +0.102 | 0.541 | 0.592 | 23.8 |
| supervoxel | 84 | 5.6 | 0.267 | 0.423 | 0.437 | 0.345 | 0.331 | 0.230 | 0.555 | 0.646 | +0.091 | 0.606 | 0.593 | 4.0 |

### Histotype F, under-segmented

| method | states | cand/state | FN-rec@1 | @3 | @8 | add prec | add prec>.5 | rem prec | start Dice | oracle-accept Dice | gain | accept-all Dice | accept-top1 Dice | time/state s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combo | 11 | 7.4 | 0.411 | 0.587 | 0.675 | 0.182 | 0.148 | 1.000 | 0.501 | 0.553 | +0.052 | 0.493 | 0.517 | 4.7 |
| ends | 11 | 3.4 | 0.043 | 0.066 | 0.067 | 0.172 | 0.094 | nan | 0.501 | 0.507 | +0.006 | 0.476 | 0.490 | 1.9 |
| intensity | 11 | 4.1 | 0.388 | 0.612 | 0.643 | 0.261 | 0.173 | nan | 0.501 | 0.551 | +0.050 | 0.507 | 0.516 | 1.2 |
| probe | 11 | 4.3 | 0.429 | 0.577 | 0.623 | 0.391 | 0.430 | 0.208 | 0.501 | 0.586 | +0.084 | 0.483 | 0.522 | 33.4 |
| supervoxel | 11 | 6.5 | 0.389 | 0.560 | 0.585 | 0.175 | 0.148 | 1.000 | 0.501 | 0.547 | +0.046 | 0.507 | 0.518 | 3.1 |

### Histotype L, under-segmented

| method | states | cand/state | FN-rec@1 | @3 | @8 | add prec | add prec>.5 | rem prec | start Dice | oracle-accept Dice | gain | accept-all Dice | accept-top1 Dice | time/state s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combo | 13 | 7.8 | 0.111 | 0.435 | 0.480 | 0.126 | 0.061 | nan | 0.330 | 0.396 | +0.066 | 0.327 | 0.309 | 10.7 |
| ends | 13 | 4.4 | 0.009 | 0.050 | 0.088 | 0.129 | 0.013 | nan | 0.330 | 0.331 | +0.000 | 0.304 | 0.304 | 4.8 |
| intensity | 13 | 3.8 | 0.111 | 0.441 | 0.444 | 0.174 | 0.137 | nan | 0.330 | 0.399 | +0.068 | 0.349 | 0.309 | 2.7 |
| probe | 13 | 5.6 | 0.360 | 0.392 | 0.444 | 0.313 | 0.297 | 0.508 | 0.330 | 0.463 | +0.132 | 0.412 | 0.450 | 39.6 |
| supervoxel | 13 | 8.0 | 0.187 | 0.364 | 0.374 | 0.149 | 0.135 | nan | 0.330 | 0.400 | +0.069 | 0.383 | 0.376 | 8.5 |

### Histotype M, under-segmented

| method | states | cand/state | FN-rec@1 | @3 | @8 | add prec | add prec>.5 | rem prec | start Dice | oracle-accept Dice | gain | accept-all Dice | accept-top1 Dice | time/state s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combo | 19 | 6.3 | 0.235 | 0.526 | 0.570 | 0.589 | 0.625 | nan | 0.630 | 0.777 | +0.147 | 0.728 | 0.665 | 10.6 |
| ends | 19 | 3.2 | 0.042 | 0.076 | 0.091 | 0.655 | 0.665 | nan | 0.630 | 0.661 | +0.030 | 0.635 | 0.631 | 5.3 |
| intensity | 19 | 4.4 | 0.259 | 0.490 | 0.506 | 0.699 | 0.688 | nan | 0.630 | 0.770 | +0.140 | 0.741 | 0.681 | 2.2 |
| probe | 19 | 4.0 | 0.327 | 0.427 | 0.447 | 0.690 | 0.754 | 0.016 | 0.630 | 0.736 | +0.105 | 0.713 | 0.730 | 37.6 |
| supervoxel | 19 | 6.0 | 0.253 | 0.504 | 0.520 | 0.590 | 0.678 | nan | 0.630 | 0.766 | +0.136 | 0.744 | 0.682 | 8.2 |

### Histotype N, under-segmented

| method | states | cand/state | FN-rec@1 | @3 | @8 | add prec | add prec>.5 | rem prec | start Dice | oracle-accept Dice | gain | accept-all Dice | accept-top1 Dice | time/state s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combo | 14 | 7.1 | 0.250 | 0.499 | 0.623 | 0.451 | 0.465 | 0.045 | 0.483 | 0.675 | +0.192 | 0.549 | 0.527 | 1.6 |
| ends | 14 | 3.8 | 0.160 | 0.248 | 0.273 | 0.490 | 0.529 | nan | 0.483 | 0.567 | +0.084 | 0.495 | 0.493 | 0.8 |
| intensity | 14 | 4.9 | 0.390 | 0.596 | 0.632 | 0.538 | 0.566 | nan | 0.483 | 0.706 | +0.223 | 0.647 | 0.669 | 0.4 |
| probe | 14 | 6.9 | 0.258 | 0.362 | 0.418 | 0.499 | 0.423 | 0.266 | 0.483 | 0.590 | +0.107 | 0.552 | 0.592 | 13.3 |
| supervoxel | 14 | 6.2 | 0.367 | 0.490 | 0.512 | 0.435 | 0.433 | 0.168 | 0.483 | 0.639 | +0.156 | 0.605 | 0.636 | 0.9 |

### Histotype V, under-segmented

| method | states | cand/state | FN-rec@1 | @3 | @8 | add prec | add prec>.5 | rem prec | start Dice | oracle-accept Dice | gain | accept-all Dice | accept-top1 Dice | time/state s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| combo | 27 | 4.4 | 0.204 | 0.359 | 0.409 | 0.315 | 0.232 | 0.162 | 0.669 | 0.729 | +0.060 | 0.594 | 0.579 | 1.3 |
| ends | 27 | 1.3 | 0.090 | 0.109 | 0.110 | 0.480 | 0.327 | nan | 0.669 | 0.675 | +0.006 | 0.656 | 0.658 | 0.5 |
| intensity | 27 | 2.7 | 0.121 | 0.286 | 0.289 | 0.584 | 0.603 | nan | 0.669 | 0.738 | +0.069 | 0.702 | 0.677 | 0.3 |
| probe | 27 | 4.7 | 0.346 | 0.478 | 0.598 | 0.333 | 0.207 | 0.088 | 0.669 | 0.759 | +0.089 | 0.499 | 0.593 | 8.0 |
| supervoxel | 27 | 3.5 | 0.215 | 0.304 | 0.312 | 0.288 | 0.203 | 0.162 | 0.669 | 0.725 | +0.056 | 0.658 | 0.643 | 0.9 |

### Per-method gain distribution (oracle-accept Dice - start Dice)

| method | share of states gaining > .02 | share losing > .02 | median gain | max gain |
|---|---|---|---|---|
| combo | 0.29 | 0.00 | +0.000 | +0.376 |
| ends | 0.10 | 0.00 | +0.000 | +0.178 |
| intensity | 0.36 | 0.00 | +0.000 | +0.324 |
| probe | 0.35 | 0.00 | +0.000 | +0.512 |
| supervoxel | 0.25 | 0.00 | +0.000 | +0.420 |

Skipped states (no GT overlap): phase1-01/F15, phase1-01/L43, phase1-01/N40, phase1-01/V46, phase1-02/F14, phase1-02/F15, phase1-02/L43, phase1-02/N40, phase1-02/V57, phase1-03/F14, phase1-03/F15, phase1-03/L43, phase1-03/N34, phase1-03/N40, phase1-03/V57, phase1-04/F14, phase1-04/F15, phase1-04/L43, phase1-04/N40, phase1-04/V57, phase1-05/F14, phase1-05/L43, phase1-05/N40, phase1-05/V46, phase1-05/V57, phase1-06-high/F14, phase1-06-high/F15, phase1-06-high/N01, phase1-06-high/N40, phase1-07-xhigh/L33, phase1-07-xhigh/N40, phase1-09-xhigh-ex/L33, phase1-09-xhigh-ex/N40
