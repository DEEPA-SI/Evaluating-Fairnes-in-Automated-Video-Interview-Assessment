# Stronger-ridge exploratory MIT comparison

The 75 deterministic fits compare a fixed alpha-1000 linear scorer selected by the existing development procedure with proxy balancing and three full-training-fold W1 strengths. All strengths are reported; the perceived attributes did not select an arm. This follow-up is exploratory because the original MIT audit had already been inspected.

Scores use the normalised rating scale. Multiply differences by six for the original 1–7 rating scale. Perceived attributes come from one observer. Skin tone retains its five ordinal categories. Categories with fewer than ten phase-matched people remain in the counts and do not enter disparity estimates.

Small nonzero contrasts and interval endpoints use short scientific notation where three-decimal rounding would hide their sign.

## Prediction utility

| Arm | Pooled MAE | Pooled SRCC | Post MAE | Post SRCC | Pre MAE | Pre SRCC |
|---|---:|---:|---:|---:|---:|---:|
| natural_task | 0.081 | 0.495 | 0.086 | 0.364 | 0.077 | 0.576 |
| visual_balanced_task | 0.081 | 0.491 | 0.086 | 0.361 | 0.076 | 0.568 |
| audio_balanced_task | 0.080 | 0.499 | 0.085 | 0.370 | 0.075 | 0.580 |
| visual_natural_w1_lambda-0.002 | 0.081 | 0.493 | 0.086 | 0.364 | 0.076 | 0.568 |
| visual_natural_w1_lambda-0.02 | 0.081 | 0.481 | 0.086 | 0.341 | 0.076 | 0.564 |
| visual_natural_w1_lambda-0.2 | 0.081 | 0.475 | 0.086 | 0.346 | 0.077 | 0.559 |
| visual_balanced_w1_lambda-0.002 | 0.081 | 0.488 | 0.086 | 0.356 | 0.076 | 0.565 |
| visual_balanced_w1_lambda-0.02 | 0.080 | 0.474 | 0.086 | 0.341 | 0.075 | 0.542 |
| visual_balanced_w1_lambda-0.2 | 0.081 | 0.446 | 0.086 | 0.321 | 0.076 | 0.522 |
| audio_natural_w1_lambda-0.002 | 0.081 | 0.491 | 0.086 | 0.357 | 0.077 | 0.573 |
| audio_natural_w1_lambda-0.02 | 0.082 | 0.462 | 0.086 | 0.306 | 0.078 | 0.544 |
| audio_natural_w1_lambda-0.2 | 0.096 | 0.173 | 0.097 | 0.061 | 0.095 | 0.194 |
| audio_balanced_w1_lambda-0.002 | 0.080 | 0.496 | 0.085 | 0.362 | 0.075 | 0.579 |
| audio_balanced_w1_lambda-0.02 | 0.080 | 0.472 | 0.086 | 0.320 | 0.075 | 0.550 |
| audio_balanced_w1_lambda-0.2 | 0.097 | 0.173 | 0.100 | 0.036 | 0.094 | 0.202 |

Solver status: converged: 62; finite_with_termination: 13.

Finite fits without reported solver convergence occur at strength(s) 0.2. Their predictions remain in the complete grid; they do not establish a converged or globally optimal strong-penalty solution.

## Proxy mechanism at strength 0.02

W1 values below are means of the five fold-specific distances. Training and held-out proxy groups use each fold's own fitted proxy; cluster numbers are never stitched across folds.

| Arm | Training W1 | Matched training W1 | Held-out W1 | Matched held-out W1 |
|---|---:|---:|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.011 | 0.036 | 0.040 | 0.047 |
| visual_balanced_w1_lambda-0.02 | 0.012 | 0.036 | 0.040 | 0.046 |
| audio_natural_w1_lambda-0.02 | 0.088 | 0.127 | 0.068 | 0.090 |
| audio_balanced_w1_lambda-0.02 | 0.094 | 0.129 | 0.073 | 0.093 |

## Primary paired comparisons at strength 0.02

Intervals use 10,000 participant bootstrap draws and are conditional on the fitted models. A positive disparity gain means lower disparity than the exact matched control. Negative MAE change and positive SRCC change mean better prediction. The intervals are exploratory; no significance or causal claim is made.

| Arm | Pooled MAE change [95% interval] | Pooled SRCC change [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | -3.8e-4 [-0.003, 0.002] | -0.013 [-0.054, 0.024] |
| visual_balanced_w1_lambda-0.02 | -0.001 [-0.003, 0.002] | -0.017 [-0.053, 0.014] |
| audio_natural_w1_lambda-0.02 | 0.001 [-0.002, 0.004] | -0.032 [-0.068, 0.003] |
| audio_balanced_w1_lambda-0.02 | 9.0e-5 [-0.003, 0.003] | -0.027 [-0.058, 0.003] |

## Post perceived-group audit

Phase-matched annotation coverage: 69 people.

### perceived gender presentation

All category counts: feminine_presentation: 43; masculine_presentation: 26.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.032 | 0.019 [-4.3e-4, 0.024] |
| visual_balanced_w1_lambda-0.02 | 0.034 | 0.016 [3.4e-4, 0.021] |
| audio_natural_w1_lambda-0.02 | 0.046 | 0.005 [-0.001, 0.012] |
| audio_balanced_w1_lambda-0.02 | 0.047 | 0.005 [-0.001, 0.011] |
### perceived racialised appearance

All category counts: unknown: 4; black_appearance_source_category: 5; east_asian_appearance_source_category: 16; middle_eastern_or_north_african_appearance_source_category: 11; south_asian_appearance_source_category: 7; southeast_asian_appearance_source_category: 2; white_appearance_source_category: 24.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.041 | -0.009 [-0.017, 0.011] |
| visual_balanced_w1_lambda-0.02 | 0.040 | -0.006 [-0.016, 0.009] |
| audio_natural_w1_lambda-0.02 | 0.030 | 0.002 [-0.001, 0.017] |
| audio_balanced_w1_lambda-0.02 | 0.030 | 0.003 [-1.4e-4, 0.015] |
### perceived skin tone

All category counts: 1: 20; 2: 27; 3: 15; 4: 2; 5: 5.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.054 | -0.001 [-0.007, 0.009] |
| visual_balanced_w1_lambda-0.02 | 0.054 | -0.003 [-0.008, 0.008] |
| audio_natural_w1_lambda-0.02 | 0.049 | 0.004 [3.9e-4, 0.014] |
| audio_balanced_w1_lambda-0.02 | 0.049 | 0.004 [-3.7e-4, 0.012] |

## Pre perceived-group audit

Phase-matched annotation coverage: 57 people.

### perceived gender presentation

All category counts: feminine_presentation: 36; masculine_presentation: 21.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.019 | 0.009 [-0.011, 0.019] |
| visual_balanced_w1_lambda-0.02 | 0.019 | 0.007 [-0.011, 0.017] |
| audio_natural_w1_lambda-0.02 | 0.022 | 0.006 [0.001, 0.013] |
| audio_balanced_w1_lambda-0.02 | 0.024 | 0.005 [0.001, 0.011] |
### perceived racialised appearance

All category counts: unknown: 2; black_appearance_source_category: 5; east_asian_appearance_source_category: 11; middle_eastern_or_north_african_appearance_source_category: 9; south_asian_appearance_source_category: 5; southeast_asian_appearance_source_category: 6; white_appearance_source_category: 19.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.027 | -0.005 [-0.012, 0.011] |
| visual_balanced_w1_lambda-0.02 | 0.027 | -0.004 [-0.010, 0.010] |
| audio_natural_w1_lambda-0.02 | 0.019 | 0.004 [-3.3e-4, 0.015] |
| audio_balanced_w1_lambda-0.02 | 0.019 | 0.005 [-9.5e-5, 0.015] |
### perceived skin tone

All category counts: 1: 5; 2: 28; 3: 16; 4: 5; 5: 3.

| Arm | Max pairwise W1 | Gain vs matched control [95% interval] |
|---|---:|---:|
| visual_natural_w1_lambda-0.02 | 0.021 | 0.008 [-0.003, 0.012] |
| visual_balanced_w1_lambda-0.02 | 0.022 | 0.008 [-0.005, 0.012] |
| audio_natural_w1_lambda-0.02 | 0.024 | 0.006 [-3.5e-4, 0.014] |
| audio_balanced_w1_lambda-0.02 | 0.024 | 0.006 [0.001, 0.013] |

## Interpretation boundary

The aggregate JSON contains every strength, fold-local proxy mechanism, group error, sparse-category count and pre-session sensitivity. Maximum pairwise W1 describes prediction distributions, while group residuals describe prediction errors. Neither establishes hiring fairness, demographic identity or compliance. The pre-session sensitivity uses the 57 available annotations without filling the twelve missing rows. Participant bootstrap uncertainty does not estimate variability from retraining or an independent population.
