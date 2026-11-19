# Published evidence and regenerated reports

The original aggregate bytes are retained. `make reproduce` derives new CSVs, figures
and a source receipt without changing these files or rerunning a fit/bootstrap.

## Complete controlled comparisons

| File | Evidence |
|---|---|
| [MIT alpha 1](study2-compact-2026-10-21-aggregate.json) | All 15 original compact arms, 75 fold fits, utility, proxy response and phase-aligned perceived-group audit |
| [MIT alpha 1000](mit-strong-ridge-2026-11-19-v1.json) | All 15 stronger-reference arms, 75 fold fits and the same controlled audit design |
| [AVI comparison](avi-compact-reconciled-v1.json) | All 15 arms, official test coverage, group metrics and primary paired intervals |
| [AVI baseline](avi-baseline-2026-11-02-reconciled.json) | Independent-corpus roster, validation selection and the descriptive predictive baseline |
| [Development-selected ridge](study2_rq1_aggregate_v1.json) | Earlier outcome-only selection, including the five alpha-1000 choices |

Each compact MIT comparison retains 62 converged fits and 13 finite high-strength
terminations. All primary-strength MIT fits converge. The AVI aggregate retains overall
nine/six solver counts without supplying individual-arm convergence status.

Readable original comparison summaries are also provided:
[MIT alpha 1](study2-compact-2026-10-21.md),
[MIT alpha 1000](mit-strong-ridge-2026-11-19-v1.md) and
[AVI](avi-compact-reconciled-v1.md). Full-precision JSON is authoritative for rounding,
units and small interval endpoints.

## Original neural and supplementary evidence

| File | Evidence |
|---|---|
| [Neural and reference utility](stage_a_utility.csv) | Thirty aggregate model/seed utility rows |
| [Paired neural utility](stage_a_secondary_utility.json) | Exposure-matched penalty effects and recorded participant-bootstrap intervals |
| [Neural diagnostics](stage_a_diagnostics_v2.json) | Checkpoint, prediction-level and reference diagnostics in aggregate form |
| [Fixed-model masks](stage_a_modality_masks.json) | Inference-only modality-dependence summaries |
| [Original perceived-group audit](original-neural-audit.json) | Aggregate original neural/group audit, with phase coverage and supported-category summaries |
| [Corrected proxy association](proxy-association-correction-2026-11-06.json) | Fold-aware proxy correspondence; visual perceived-gender NMI and ARI equal 1 |
| [Descriptive expansion](thesis-expanded-analysis-2026-11-17.json) | Group errors, spread, objective components, fold responses and interactions derived from aggregates |
| [Explanatory figure values](thesis-explanatory-figure-data-2026-11-17.json) | Exact source values and units for the original explanatory figures |

`original-neural-audit.json` is a byte-for-byte public copy of the original Stage-C
aggregate, not participant-level predictions. Historical references within derivative
receipts identify their original source locations; they are not dependencies of the
public reporting command. Superseded defective exploratory outputs are not included.

## Included example and fresh calculation

[reproduction/](reproduction/) contains the generated example: the full 45-row compact
grid, 112 primary matched-contrast rows, three SVG/PNG figures, a readable summary and
source receipt. These are reporting transformations of the evidence above.

Regenerate them into a new directory:

```bash
make reproduce OUTPUT_DIR=build/reproduction
```

The [reproduction guide](../docs/reproduction.md) describes file schemas, score scales,
contrast signs and what is calculated. The publication manifest verifies the exact
distributed file identities. No individual annotation, feature vector or prediction
row is distributed.
