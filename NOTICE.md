# Attribution and release scope

This repository accompanies Deepa Singh's *Evaluating Fairness in Automated Video
Interview Assessment*, Master of Applied Technologies, Unitec Institute of Technology.
Principal supervisor: Ali Keivanmarz. Co-supervisor: Professor Hamid Sharifzadeh.

The project-authored source code retains Deepa Singh's MIT licence. The release
preserves the working numerical implementations and accepted aggregate observations;
public reporting and portability helpers are identified by the source-export record.

## Research and implementation attribution

- MIT Interview and its aggregated interview ratings: Naim et al., *Automated Analysis
  and Prediction of Job Interview Performance* (2018).
- AVI-Personality and its recruiter ratings/self-reported attributes:
  [APAL-SEU/AVI-Personality](https://huggingface.co/datasets/APAL-SEU/AVI-Personality).
- Visual representations: Oquab et al., DINOv2; audio representations: Chen et al.,
  WavLM; text representations: Song et al., MPNet, and Reimers and Gurevych,
  Sentence-BERT/Sentence Transformers.
- The adversarial reweighting comparator follows the approach of Lahoti et al.;
  pretrained-model encoders and external numerical libraries retain their own attribution.

No claim is made that this study invented Wasserstein distance, clustering, ridge
regression or the cited model families. Its contribution is the controlled empirical
evaluation, proxy/audit relationship and comparison across the completed settings.

## Included and omitted material

The project licence applies to project-authored software and documentation, not to
third-party datasets, pretrained weights or external libraries. No raw recordings,
transcripts, crops, embeddings, model weights, individual predictions or annotation
rows are redistributed. Aggregate results retain their original scientific meaning
and precision. Dataset and model terms remain those supplied by their providers.

This clean release does not include private development history, administrative
coordination records, credentials or superseded defective experimental outputs.
