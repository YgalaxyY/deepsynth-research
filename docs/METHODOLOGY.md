# Experimental protocol

The investigation is quality-aware fusion for audiovisual manipulation detection. The visual encoder is an ImageNet-pretrained EfficientNet-B4 feature extractor, not a newly trained deepfake backbone. Two small MLP heads produce branch scores. A fixed sigmoid of a frame sharpness proxy combines them.

## Current controls

The sampler uses seed 42. Clip labels are genuine audiovisual recordings (0) versus manipulated video and audio (1). Encoded variants preserve the selected filename and source-group ID. A deterministic group holdout is saved before training. Evaluation requires all held-out filenames at each quality; a missing clip aborts the comparison.

The single scalar quality proxy is `clip(variance(Laplacian(gray_frame)) / 500, 0.01, 1)`. Its normalization and gate constants are hand chosen. They must be selected using training/validation data, never the final test set. A future experiment should use train/validation/test splits and multiple seeds. The current fixed 30-epoch run does not tune against the test set.

## Interpretation

The visual-branch score is a paired diagnostic control, not an independently optimized benchmark. Improvements would require independent visual/audio baselines, fixed fusion, learned fusion, repeated seeds, confidence intervals and the mixed manipulation categories. Compression quality and resize resolution change together, so the experiment cannot isolate their causal effects.

Directory IDs do not establish complete independence of every swapped target identity. Before a publication-grade benchmark, inspect the dataset metadata and enforce identity-level connected components across all source and target people, or use an appropriate published split.

No conclusion about a person's authenticity, wrongdoing or identity should be drawn from this research prototype alone.
