# AVI compact fairness comparison

This is an exploratory intervention follow-up on an independent interview corpus. The earlier AVI prediction/audit results had already been inspected. All fifteen settings are reported; no strength is selected from test or gender-audit results.

Sampled full-frame visual and clip-audio features are pooled over six clips per participant; these are not MIT candidate-face crops. The official test contains 130 people. Target errors below use the original 1–5 rating scale; score W1 uses the normalised prediction scale. Source gender categories are self-reports, not MIT's perceived-appearance labels.

Selected ridge alpha: 1000 (validation MAE only).
Solver dispositions: {'converged': 9, 'finite_with_termination': 6}.

| Arm | Test MAE | SRCC | Gender score W1 |
|---|---:|---:|---:|
| natural_task | 0.419 | 0.473 | 0.062 |
| visual_balanced_task | 0.420 | 0.463 | 0.064 |
| visual_natural_w1_lambda-0.002 | 0.419 | 0.473 | 0.061 |
| visual_natural_w1_lambda-0.02 | 0.421 | 0.455 | 0.061 |
| visual_natural_w1_lambda-0.2 | 0.424 | 0.445 | 0.059 |
| visual_balanced_w1_lambda-0.002 | 0.419 | 0.464 | 0.064 |
| visual_balanced_w1_lambda-0.02 | 0.417 | 0.464 | 0.065 |
| visual_balanced_w1_lambda-0.2 | 0.420 | 0.447 | 0.050 |
| audio_balanced_task | 0.421 | 0.467 | 0.068 |
| audio_natural_w1_lambda-0.002 | 0.419 | 0.473 | 0.061 |
| audio_natural_w1_lambda-0.02 | 0.422 | 0.442 | 0.057 |
| audio_natural_w1_lambda-0.2 | 0.453 | 0.295 | 0.050 |
| audio_balanced_w1_lambda-0.002 | 0.421 | 0.466 | 0.068 |
| audio_balanced_w1_lambda-0.02 | 0.424 | 0.435 | 0.063 |
| audio_balanced_w1_lambda-0.2 | 0.462 | 0.320 | 0.054 |

## Primary paired comparison

Visual natural-exposure W1 at 0.02 versus its task-only control: MAE change 0.002 [-0.003, 0.008] rating points; SRCC change -0.018 [-0.041, 0.002]; disparity gain 0.001 [-0.002, 0.004].

Intervals use 10,000 paired participant bootstrap draws and are pointwise, conditional on these fitted models. They do not estimate retraining uncertainty or establish employment validity, compliance or universal mitigation.

## Source groups and signed error gap

Test source-code counts: {'1': 67, '2': 62, '3': 0, '4': 1, 'unknown': 0}.

| Arm | Signed MAE gap, code 1 minus code 2 [95% interval] |
|---|---:|
| natural_task | 0.067 [-0.054, 0.181] |
| visual_natural_w1_lambda-0.02 | 0.066 [-0.058, 0.182] |

The absolute gap is descriptive; a positive interval for its absolute value is not evidence that the signed group difference excludes zero. Sparse and nonresponse source categories remain in counts without being merged.

Some bounded solver fits terminated without reported convergence. Their finite predictions remain in the complete grid and are explicitly qualified; no global-optimum claim is made.
