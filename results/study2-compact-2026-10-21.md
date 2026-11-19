# Compact exploratory MIT follow-up

The 75 deterministic fits compare a fixed alpha-1 linear scorer with proxy balancing and three full-training-fold W1 strengths. All strengths are reported; the perceived attributes did not select an arm. This follow-up is exploratory because the original MIT audit had already been inspected.

Scores use the normalised rating scale. Multiply differences by six for the original 1–7 rating scale. Perceived attributes come from one observer. Skin tone retains its five ordinal categories. Categories with fewer than ten phase-matched people remain in the counts and do not enter disparity estimates.

## Prediction utility

| Arm | Pooled MAE | Pooled SRCC | Post MAE | Post SRCC | Pre MAE | Pre SRCC |
|---|---:|---:|---:|---:|---:|---:|
| natural_task | 0.089 | 0.418 | 0.095 | 0.276 | 0.083 | 0.502 |
| visual_balanced_task | 0.089 | 0.418 | 0.095 | 0.276 | 0.083 | 0.502 |
| audio_balanced_task | 0.089 | 0.418 | 0.095 | 0.276 | 0.083 | 0.502 |
| visual_natural_w1_lambda-0.002 | 0.089 | 0.416 | 0.095 | 0.279 | 0.083 | 0.497 |
| visual_natural_w1_lambda-0.02 | 0.090 | 0.389 | 0.096 | 0.256 | 0.083 | 0.463 |
| visual_natural_w1_lambda-0.2 | 0.090 | 0.383 | 0.096 | 0.231 | 0.084 | 0.476 |
| visual_balanced_w1_lambda-0.002 | 0.089 | 0.416 | 0.095 | 0.281 | 0.083 | 0.495 |
| visual_balanced_w1_lambda-0.02 | 0.090 | 0.387 | 0.096 | 0.259 | 0.084 | 0.452 |
| visual_balanced_w1_lambda-0.2 | 0.089 | 0.392 | 0.095 | 0.240 | 0.083 | 0.487 |
| audio_natural_w1_lambda-0.002 | 0.089 | 0.414 | 0.095 | 0.268 | 0.084 | 0.500 |
| audio_natural_w1_lambda-0.02 | 0.091 | 0.374 | 0.096 | 0.212 | 0.085 | 0.468 |
| audio_natural_w1_lambda-0.2 | 0.103 | 0.134 | 0.103 | 0.037 | 0.102 | 0.150 |
| audio_balanced_w1_lambda-0.002 | 0.089 | 0.414 | 0.095 | 0.269 | 0.083 | 0.499 |
| audio_balanced_w1_lambda-0.02 | 0.089 | 0.385 | 0.096 | 0.231 | 0.083 | 0.473 |
| audio_balanced_w1_lambda-0.2 | 0.103 | 0.129 | 0.106 | 0.018 | 0.099 | 0.178 |

Solver status: converged: 62; finite_with_termination: 13.

Finite fits without reported solver convergence occur at strength(s) 0.2. Their predictions remain in the complete grid; they do not establish a converged or globally optimal strong-penalty solution.

## Proxy mechanism at strength 0.02

W1 values below are means of the five fold-specific distances. Training and held-out proxy groups use each fold's own fitted proxy; cluster numbers are never stitched across folds.

| Arm | Training W1 | Matched training W1 | Held-out W1 | Matched held-out W1 |
|---|---:|---:|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.009 | 0.036 | 0.035 | 0.045 |
| visual_balanced_w1_lambda-0.02 | 0.010 | 0.036 | 0.036 | 0.045 |
| audio_natural_w1_lambda-0.02 | 0.090 | 0.134 | 0.068 | 0.087 |
| audio_balanced_w1_lambda-0.02 | 0.096 | 0.134 | 0.070 | 0.087 |

## Primary paired comparisons at strength 0.02

Intervals use 10,000 participant bootstrap draws and are conditional on the fitted models. A positive disparity gain means lower disparity than the exact matched control. Negative MAE change and positive SRCC change mean better prediction. The intervals are exploratory; no significance or causal claim is made.

| Arm | Pooled MAE change [95% interval] | Pooled SRCC change [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.001 [-0.002, 0.003] | -0.030 [-0.069, 0.008] |
| visual_balanced_w1_lambda-0.02 | 0.001 [-0.002, 0.003] | -0.031 [-0.066, 0.004] |
| audio_natural_w1_lambda-0.02 | 0.002 [-0.001, 0.005] | -0.044 [-0.080, -0.008] |
| audio_balanced_w1_lambda-0.02 | 0.000 [-0.003, 0.003] | -0.033 [-0.067, 0.002] |

## Post perceived-group audit

Phase-matched annotation coverage: 69 people.

### perceived gender presentation

All category counts: feminine_presentation: 43; masculine_presentation: 26.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.029 | 0.020 [-0.005, 0.025] |
| visual_balanced_w1_lambda-0.02 | 0.029 | 0.020 [-0.004, 0.024] |
| audio_natural_w1_lambda-0.02 | 0.045 | 0.004 [-0.002, 0.011] |
| audio_balanced_w1_lambda-0.02 | 0.045 | 0.004 [-0.002, 0.010] |
### perceived racialised appearance

All category counts: unknown: 4; black_appearance_source_category: 5; east_asian_appearance_source_category: 16; middle_eastern_or_north_african_appearance_source_category: 11; south_asian_appearance_source_category: 7; southeast_asian_appearance_source_category: 2; white_appearance_source_category: 24.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.038 | -0.009 [-0.015, 0.010] |
| visual_balanced_w1_lambda-0.02 | 0.037 | -0.007 [-0.013, 0.010] |
| audio_natural_w1_lambda-0.02 | 0.032 | -0.002 [-0.006, 0.016] |
| audio_balanced_w1_lambda-0.02 | 0.032 | -0.002 [-0.005, 0.015] |
### perceived skin tone

All category counts: 1: 20; 2: 27; 3: 15; 4: 2; 5: 5.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.065 | 0.001 [-0.006, 0.010] |
| visual_balanced_w1_lambda-0.02 | 0.065 | 0.001 [-0.005, 0.010] |
| audio_natural_w1_lambda-0.02 | 0.063 | 0.003 [-0.002, 0.013] |
| audio_balanced_w1_lambda-0.02 | 0.064 | 0.002 [-0.003, 0.012] |

## Pre perceived-group audit

Phase-matched annotation coverage: 57 people.

### perceived gender presentation

All category counts: feminine_presentation: 36; masculine_presentation: 21.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.022 | 0.005 [-0.016, 0.019] |
| visual_balanced_w1_lambda-0.02 | 0.023 | 0.005 [-0.015, 0.018] |
| audio_natural_w1_lambda-0.02 | 0.023 | 0.005 [0.000, 0.013] |
| audio_balanced_w1_lambda-0.02 | 0.023 | 0.005 [0.000, 0.012] |
### perceived racialised appearance

All category counts: unknown: 2; black_appearance_source_category: 5; east_asian_appearance_source_category: 11; middle_eastern_or_north_african_appearance_source_category: 9; south_asian_appearance_source_category: 5; southeast_asian_appearance_source_category: 6; white_appearance_source_category: 19.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.035 | -0.008 [-0.013, 0.007] |
| visual_balanced_w1_lambda-0.02 | 0.033 | -0.006 [-0.012, 0.007] |
| audio_natural_w1_lambda-0.02 | 0.022 | 0.006 [-0.002, 0.015] |
| audio_balanced_w1_lambda-0.02 | 0.021 | 0.006 [-0.002, 0.016] |
### perceived skin tone

All category counts: 1: 5; 2: 28; 3: 16; 4: 5; 5: 3.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.023 | 0.005 [-0.005, 0.012] |
| visual_balanced_w1_lambda-0.02 | 0.024 | 0.004 [-0.006, 0.012] |
| audio_natural_w1_lambda-0.02 | 0.024 | 0.004 [-0.002, 0.015] |
| audio_balanced_w1_lambda-0.02 | 0.023 | 0.004 [-0.001, 0.014] |

## Interpretation boundary

The aggregate JSON contains every strength, fold-local proxy mechanism, group error, sparse-category count and pre-session sensitivity. Maximum pairwise W1 describes prediction distributions, while group residuals describe prediction errors. Neither establishes hiring fairness, demographic identity or compliance. The pre-session sensitivity uses the 57 available annotations without filling the twelve missing rows. Participant bootstrap uncertainty does not estimate variability from retraining or an independent population.
