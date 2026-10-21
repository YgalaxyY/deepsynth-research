# Release preparation

The provided archive combined research scripts, a presentation UI, dataset copies, generated features, a virtual environment and local artifacts. This source release contains the research pipeline only.

The old UI's illustrative randomized outputs are excluded. They were not model evaluation results. Historical scores are not carried into this release.

Changes made during portfolio preparation:

- Corrected genuine-class directory selection.
- Converted OpenCV BGR frames to RGB and used the pretrained encoder's transforms.
- Replaced silent zero audio with actual FFmpeg decoding and explicit errors.
- Replaced file-level random splitting with deterministic source-group holdout.
- Evaluated only held-out clips at every quality and checked complete coverage.
- Replaced a machine-specific FFmpeg location with PATH discovery.

These changes invalidate comparisons with an earlier run. Protocol unit tests and syntax checks validate the preparation changes; model performance requires a new full experiment. No datasets, private documents, personal paths, weights or old metrics are bundled.
