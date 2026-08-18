# Submission evaluation evidence

This package records a fixed-seed FinFET synthetic evaluation generated with the existing public generator and evaluated with the exact production path used by `localize.py`.

- Full evaluation pool: 40 pairs (10 per actual generator noise mode), seed base `20260818`
- Selected examples: 30 pairs, deterministically stratified by noise mode and observed error rank
- Full-pool Accuracy@1 px: 57.50%; Accuracy@5 px: 80.00%
- Selected-30 Accuracy@1 px: 60.00%; Accuracy@5 px: 80.00%

The full 40-pair source pool is intentionally not committed. The result CSV
retains reproducible relative `evidence_pool/<mode>/...` paths and the fixed
seed; the committed selected-30 metadata points to the included copies.

## Evidence files

- [Result CSV](results/localization_results.csv) and [measured metrics](results/metrics.md)
- [Environment](results/environment.md)
- [Selected pair metadata](selected_30/metadata.csv), [comments](selected_30/pair_comments.md), and overlays (green = GT; red = prediction)
- [Graphs](graphs/) including PR, tolerance, confidence-calibration matrices, noise stress, and error distribution
- [Failure analysis](failure_cases/FAILURE_ANALYSIS.md)

## Interpretation

The full-pool figure is the honest measured result for this fixed synthetic pool. It is a test of consistency with the generator's corrected centre-rule labels, not an independently labelled real-SEM benchmark. The selected 30 are deliberately diverse examples and are reported separately.
