# Aggregate research overview

This release examines prediction, learned-group score alignment and independently measured group outcomes as separate questions. Matched task-only controls distinguish the added Wasserstein penalty from balanced exposure. Every fixed strength remains in the complete comparison; no best arm is selected from the audit.

## Predictive references

| Setting | MAE in original rating points | Spearman correlation | Evaluation |
|---|---:|---:|---|
| MIT: ridge alpha 1 | 0.535 | 0.418 | pooled out of fold sessions |
| MIT: ridge alpha 1000 | 0.487 | 0.495 | pooled out of fold sessions |
| AVI: validation-selected ridge | 0.419 | 0.473 | official test |

MIT errors use a 1–7 rating scale and AVI errors a 1–5 scale; they are not pooled across corpora. MIT's pooled correlations are not the separately reported mean-fold correlation of the development-selected procedure.

## What the learned groups represent

The visual proxy partition agrees exactly with one observer's perceived gender-presentation grouping in the corrected MIT comparison (NMI 1.000; adjusted Rand index 1.000). Here, excluding annotations during fitting still leaves associated information in the representations. These measurements describe perceived presentation, not verified demographic identity.

## Primary-strength group-score comparisons

| Setting and arm | Gender score-disparity gain | Existing 95% interval |
|---|---:|---:|
| MIT: ridge alpha 1: visual natural w1 lambda-0.02 | 0.020 | [-0.005, 0.025] |
| MIT: ridge alpha 1: visual balanced w1 lambda-0.02 | 0.020 | [-0.004, 0.024] |
| MIT: ridge alpha 1: audio natural w1 lambda-0.02 | 0.004 | [-0.002, 0.011] |
| MIT: ridge alpha 1: audio balanced w1 lambda-0.02 | 0.004 | [-0.002, 0.010] |
| MIT: ridge alpha 1000: visual natural w1 lambda-0.02 | 0.019 | [-0.00043, 0.024] |
| MIT: ridge alpha 1000: visual balanced w1 lambda-0.02 | 0.016 | [0.00034, 0.021] |
| MIT: ridge alpha 1000: audio natural w1 lambda-0.02 | 0.005 | [-0.001, 0.012] |
| MIT: ridge alpha 1000: audio balanced w1 lambda-0.02 | 0.005 | [-0.001, 0.011] |
| AVI: validation-selected ridge: visual natural w1 lambda-0.02 | 0.001 | [-0.002, 0.004] |
| AVI: validation-selected ridge: visual balanced w1 lambda-0.02 | -0.00025 | [-0.003, 0.002] |
| AVI: validation-selected ridge: audio natural w1 lambda-0.02 | 0.005 | [3.4e-05, 0.011] |
| AVI: validation-selected ridge: audio balanced w1 lambda-0.02 | 0.005 | [-8e-05, 0.010] |

Positive gain denotes a smaller score-distribution distance, not necessarily more accurate prediction for every group. The CSV retains every available MIT axis and pre-session sensitivity, plus AVI's separately sourced gender audit. AVI companions are marked secondary unless its source explicitly designates the contrast primary.

## Reproduction scope

This command replays tables and figures from published aggregates. It does not extract features, train models, reconstruct participant predictions or rerun participant-bootstrap inference. Intervals are copied from the saved paired pointwise analyses, conditional on the fitted predictions. MIT attributes come from one observer; AVI uses self-reported gender. AVI's aggregate solver counts do not identify individual-arm or primary-arm convergence.
