# E16 candidate-decision replay — run `e16`

Model `anthropic/claude-sonnet-5-5`, thinking `high`, 3 reps × 27 replayed reviews × 2 conditions (base, lesion); scored 2026-10-19T04:59:04Z. pi SDK 1.3.0 (auth shim 3.4.3). Positive class = candidate labelled 'lesion' by GT (precision > 0.5).

## Caveats
- Single tool-less decision turn over replayed review images; no interaction, so the candidate sets are frozen (the agent could iterate with seg_accept/seg_view).
- GT labels are per candidate (precision > 0.5 → 'lesion'); borderline candidates (precision near 0.5) are noisy ground truth.
- The original agent decisions (`agent_accepted`) are case-level over the whole trajectory, not per review version — shown in the per-review table for reference only.

## Headline (pooled over reviews, reps averaged for Dice)

| condition | n calls | sens | spec | mean accepted/review | mean Dice Δ vs reviewed | oracle Dice Δ | parse fail | invalid ids | resp error |
|---|---|---|---|---|---|---|---|---|---|
| base | 81 | 0.056 | 0.988 | 0.07 | -0.003 | 0.037 | 0 | 0 | 0 |
| lesion | 81 | 0.130 | 0.988 | 0.12 | -0.000 | 0.037 | 0 | 0 | 0 |

## Per histotype (sens / spec / mean Dice Δ)

| histotype | base sens | base spec | base Dice Δ | lesion sens | lesion spec | lesion Dice Δ | n reviews |
|---|---|---|---|---|---|---|---|
| F | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 6 |
| L | 0.000 | 1.000 | -0.000 | 0.000 | 1.000 | -0.000 | 7 |
| M | 0.000 | 1.000 | -0.000 | 0.111 | 1.000 | 0.010 | 4 |
| N | 0.200 | 1.000 | 0.001 | 0.333 | 1.000 | 0.005 | 5 |
| V | 0.000 | 0.929 | -0.015 | 0.000 | 0.929 | -0.015 | 5 |

## Per-review table

| key | h | seq | cands (lesion) | dice reviewed | oracle Δ | base reps (accepted, Δ) | lesion reps (accepted, Δ) | agent accepted (case-level, reference) |
|---|---|---|---|---|---|---|---|---|
| F14_v1 | F | T2 | 1 (0) | 0.722 | -0.000 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| F15_v3 | F | T2 | 5 (0) | 0.006 | -0.000 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| F32_v4 | F | T2 | 3 (0) | 0.430 | +0.000 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| F44_v4 | F | T2 | 5 (2) | 0.734 | +0.117 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| F49_v1 | F | T2 | 3 (0) | 0.806 | +0.000 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| F50_v1 | F | T2 | 5 (2) | 0.880 | +0.027 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| L19_v1 | L | T1 | 3 (0) | 0.732 | +0.000 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| L19_v4 | L | T1 | 3 (0) | 0.831 | -0.000 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| L20_v1 | L | T1 | 5 (0) | 0.941 | +0.000 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| L33_v2 | L | T2 | 4 (0) | 0.000 | +0.000 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| L35_v3 | L | T1 | 4 (0) | 0.903 | +0.000 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| L38_v1 | L | T1 | 2 (0) | 0.967 | +0.000 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| L43_v1 | L | T1 | 5 (1) | 0.669 | +0.154 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| M01_v1 | M | T2 | 5 (0) | 0.892 | -0.000 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| M08_v1 | M | T2 | 3 (3) | 0.736 | +0.096 | none (+0.000); none (+0.000); none (+0.000) | C1 (+0.061); C1 (+0.061); none (+0.000) | – |
| M21_v2 | M | T2 | 5 (0) | 0.834 | -0.000 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| M33_v4 | M | T2 | 5 (3) | 0.634 | +0.189 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| N01_v1 | N | T2 | 1 (0) | 0.000 | +0.000 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| N18_v3 | N | T2 | 5 (0) | 0.865 | -0.000 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| N21_v2 | N | T2 | 2 (2) | 0.853 | +0.038 | C2 (+0.006); C2 (+0.006); C2 (+0.006) | C2 (+0.006); C1,C2 (+0.038); C1,C2 (+0.038) | – |
| N26_v1 | N | T2 | 5 (0) | 0.929 | -0.000 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| N34_v3 | N | T2 | 5 (3) | 0.439 | +0.264 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| V37_v1 | V | T2 | 1 (1) | 0.766 | +0.044 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |
| V46_v3 | V | T2 | 3 (0) | 0.854 | -0.000 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | – |
| V57_v1 | V | T2 | 4 (0) | 0.699 | -0.000 | none (-0.000); none (-0.000); none (-0.000) | none (-0.000); none (-0.000); none (-0.000) | C2 |
| V57_v2 | V | T2 | 5 (0) | 0.656 | -0.000 | C4 (-0.075); C4 (-0.075); C4 (-0.075) | C4 (-0.075); C4 (-0.075); C4 (-0.075) | C2 |
| V65_v1 | V | T2 | 3 (1) | 0.647 | +0.080 | none (+0.000); none (+0.000); none (+0.000) | none (+0.000); none (+0.000); none (+0.000) | – |

## Cost / latency

- 162 calls, total cost ≈ **$1.27**, mean latency 3.5s

## Scoring rules
- Last line strict `ACCEPT: none` or `ACCEPT: C1, C2` (`none` case-insensitive, ids uppercase `C<number>`); anything else = `parse_fail`.
- Ids not in the review = `invalid_ids` (counted; no confusion counts, no Dice).
- dice_after = 2·(mask_gt_vox + Σ inter_vox) / (mask_vox + Σ region_vox + gt_vox); candidates are disjoint and outside the mask (asserted at prepare time).
- Mean Dice Δ: per review mean over reps, then mean over reviews.
- Oracle Δ: accept exactly the lesion-labelled candidates (condition-independent).
