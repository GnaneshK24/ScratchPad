# Documented high-error cases

All cases below are genuine, frozen-localizer measurements from the selected
30-pair subset. Green marks ground truth and red marks the prediction.

| Case | Sample | Error | Confidence | Generator metadata |
| --- | --- | ---: | ---: | --- |
| [01](failure_case_01.png) | `medium_001` | 319.130 px | 0.000000 | medium noise; 0.00° rotation |
| [02](failure_case_02.png) | `medium_000` | 290.583 px | 0.000000 | medium noise; 0.00° rotation |
| [03](failure_case_03.png) | `high_000` | 216.479 px | 0.000000 | high noise; 2.71° rotation |
| [04](failure_case_04.png) | `high_008` | 106.345 px | 0.000000 | high noise; -0.78° rotation |

These are failures at the 5 px criterion. See
[WHY_FAILURES_OCCUR.md](WHY_FAILURES_OCCUR.md) for the evidence-based cause
analysis and limitations of that interpretation.
