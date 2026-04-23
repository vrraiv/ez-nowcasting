# Notebook Plan

This directory is reserved for exploratory notebooks that sit alongside the source scaffold.

Suggested notebook sequence:

1. `01_dataset_discovery.ipynb`
2. `02_download_and_qc.ipynb`
3. `03_transforms_and_targets.ipynb`
4. `04_baseline_model_prototypes.ipynb`

Keep notebooks thin and promote reusable logic into `src/data_access/`, `src/transforms/`, `src/features/`, `src/models/`, and `src/evaluation/` as soon as code stabilizes.
