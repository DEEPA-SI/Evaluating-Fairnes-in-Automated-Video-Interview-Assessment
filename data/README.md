# Data and representation guide

The default public report workflow uses the aggregate result files already included
in this repository. No dataset download or account is needed for `make all`.

## Study datasets

| Setting | Analysis unit and prediction target | Evaluation and audit |
|---|---|---|
| MIT Interview | 69 people, 138 paired pre/post interviews; crowd-perceived hiring recommendation, originally 1–7 | Five participant-grouped folds; 69 post-session annotations and a 57-person pre-session sensitivity |
| AVI-Personality | Six recorded responses per person; recruiter-rated interview performance, originally 1–5 | Official participant-separated roles: 450 supervised training, 64 validation and 130 test participants |

MIT uses one observer's perceived gender presentation, racialised appearance and skin
tone. These are perceptions of recordings, not verified demographic identities. Skin
tone retains five ordinal categories. AVI uses self-reported gender; it supplies no
equivalent skin-tone audit. The corpora have different rating constructs and are not
pooled into a single performance estimate.

The AVI source is [APAL-SEU/AVI-Personality](https://huggingface.co/datasets/APAL-SEU/AVI-Personality).
The MIT source is described by Naim et al., *Automated Analysis and Prediction of Job
Interview Performance*, IEEE Transactions on Affective Computing (2018),
[doi:10.1109/TAFFC.2016.2614299](https://doi.org/10.1109/TAFFC.2016.2614299).
Use the providers' original distributions and terms when obtaining source data.

## Included public evidence

Only reviewed aggregate outputs are included: complete utility/disparity grids,
paired interval summaries, fold-level solver diagnostics, group counts, proxy
correspondence summaries and reporting derivatives. The [result index](../results/README.md)
identifies each input. Some original hash keys name historical source inputs; they
are provenance bindings, not files needed for public report regeneration.

Raw video/audio, transcripts, face crops, participant embeddings, model checkpoints,
individual predictions and annotation rows are not part of this release. They are
not downloaded implicitly by public commands.

## Inputs for source-dependent reproduction

The compact numerical implementation consumes accepted cached representations, not
raw video. MIT's ordered matrix has 138 rows and 3,328 coordinates: DINOv2-large visual
means (1,024), WavLM-Base+ audio means (1,536) and MPNet text means (768). Both sessions
of a participant stay together. The accepted feature closure, complete-pair eligibility,
target bindings and sidecars preserve the same input roster and target scaling.

AVI uses participant-level means of six question summaries, giving 2,560 coordinates
from visual and audio representations. Separate outcome and audit files prevent gender
from entering fitting or selection. Accepted manifests and protocol checks bind the
source release. Neither source-dependent runner substitutes synthetic features or
guesses missing annotations.

For a licensed local source release, provide the following to the optional commands:

- MIT: the accepted feature directory; Stage-A target/eligibility closure directory;
  and the separate source-annotation release for auditing.
- AVI: the participant-feature manifest and shards; separated participant outcomes;
  and separated audit attributes. The fixed protocol records their original identities.
- Raw-media reconstruction: source recordings/transcripts, the participant/session
  mapping, approved face-selection metadata and pinned encoder artifacts. The feature
  and preprocessing code is included; no model weights are vendored.

The [reproduction guide](../docs/reproduction.md) provides explicit argument examples.
These optional inputs are not prerequisites for exploring or verifying the public
aggregate evidence.
