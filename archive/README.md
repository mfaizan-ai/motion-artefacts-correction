# Archive

Old scripts kept for reference, no longer compatible with the current model/dataset.

## test.py

Inference/evaluation script for the **plain** `DisentangledCycleGAN` (`models/model.py`)
trained on the older `cyclegans_dataset` (PSC-normalized, `dataset.py`'s `ValFMRIDataset`).

Not compatible with the current `SpatioTemporalCycleGAN` (`models/st_model.py`) models
(`st_v3_ddp_with_roi_time_series_cycleGANS`, `st_v4_ddp_disc_temporal_roi`), which use a
different architecture (factorized R(3+1)D blocks, different spatial dims/state_dict keys)
and a different dataset (`grade_dataset.py`'s `FMRIUnpairedGradeDataset`, robust_p5p95
normalization on `motion_grades_chunk_5_dataset_hfiltered`). Loading an ST-model checkpoint
into `DisentangledCycleGAN` fails outright (state_dict key mismatch).

Superseded by `test.py` in the project root, which targets the current ST models.
