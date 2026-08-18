# Why these failures occur

## What the measured evidence shows

The four documented failures have errors from 106.345 px to 319.130 px and
all have a reported confidence of `0.0`. The red prediction and green ground
truth boxes in each overlay are visibly separated, so these are genuine
wrong-region localizations rather than rounding or centre-coordinate errors.

Two failures were generated with the `medium` profile and no rotation; the
other two use the `high` profile, with rotations of +2.71 degrees and -0.78
degrees. This establishes that severe errors are not restricted to rotated
images, although high-noise rotation is an additional challenge in cases 03
and 04.

## Likely failure mechanism

The frozen production method is a classical appearance matcher. FinFET layouts
contain repeated, visually similar structures, so several search locations can
look plausible for a reference crop. Noise, blur, contrast variation, and—in
the high profile—small rotation reduce the image detail that distinguishes
those repetitions. In these cases the selected visual candidate is a distant
periodic alternative rather than the centre-rule ground-truth location.

The zero confidences are consistent with ambiguity rather than proof of a
high-certainty wrong answer. They should not be treated as a calibrated
probability.

## Scope and limitation

The evidence records final predictions, errors, confidence, noise profile, and
rotation; it does not retain a complete per-candidate trace. Therefore this
analysis identifies the observed periodic-ambiguity mechanism supported by
the images and metadata, but does not claim a unique internal cause for every
wrong ranking. No matcher setting, candidate-generation step, ranking rule,
centre tie-break, or ground-truth label was changed to remove these failures.

Additional independent context—such as a larger reference field, a less
repetitive feature, or external stage/location information—could disambiguate
the competing regions without relabelling the image.
