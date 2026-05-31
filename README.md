# CycleGAN-Based fMRI Motion Artefact Correction

motion artefact correction is treated as an **unpaired image-to-image translation task**. Motion-corrupted chunks and low-motion chunks represent two separate domains, and a CycleGAN-like model learns to map corrupted fMRI data toward the motion-free distribution. The architecture aims to separate motion artefacts from the underlying BOLD signal content, while preserving spatial structure and temporal consistency. Training is guided using adversarial, cycle-consistency, identity, and structure-preserving losses.

## Motion corrupted data selection from the video state fMRI 

This step identifies fMRI time-series chunks that contain sustained head-motion artefacts and prepares them as the **motion-corrupted domain** for CycleGAN training.

### Overview

For each video-task BOLD run, framewise displacement (FD) is calculated from the corresponding motion-parameter file. A sliding window is then moved across the FD time series to identify chunks that contain meaningful and sustained motion contamination.

Each accepted chunk is saved as a row in the output CSV, including the subject/session/run information, BOLD file path, chunk start and end volumes, and FD-based motion statistics.

### Selection Criteria

A chunk is labelled as motion-corrupted when it satisfies the following conditions:

- A sufficient fraction of volumes exceed the high-motion FD threshold.
- The chunk contains a sustained motion event across consecutive volumes.
- The mean FD of the chunk indicates meaningful motion severity.
- Extremely large, catastrophic FD values are excluded using a maximum-FD guard.

### High motion chunk selection criteria:

| Parameter | Default | Description |
|---|---:|---|
| `chunk_size` | `20` | Number of volumes per chunk |
| `step` | `1` | Sliding-window step size |
| `thr_high` | `1.0 mm` | FD threshold for high-motion volumes |
| `min_frac_high` | `0.5` | Minimum fraction of high-motion volumes |
| `min_sustained` | `3` | Minimum consecutive high-motion volumes |
| `min_mean_fd` | `1.0 mm` | Minimum mean FD required for the chunk |
| `max_fd` | `10.0 mm` | Upper FD limit to remove catastrophic artefacts |

The script generates a CSV file containing all selected corrupted chunks:

```text
subject_id, session_id, run_id, task,
bold_file, motion_parameter_file,
chunk_start, chunk_end,
chunk_mean_fd, chunk_max_fd,
n_vols_above_thr, frac_vols_above_thr,
max_sustained_streak, n_total_vols_in_run
```

## Low-motion data selection from the video and resting state fMRI 

This step extracts low-motion fMRI chunks that are suitable for the **motion-free / clean domain** in CycleGAN training.

### Overview

For each fMRI run, framewise displacement (FD) is calculated from the corresponding motion-parameter file. Volumes with FD above the low-motion threshold are treated as motion spikes. To avoid selecting volumes close to motion events, a temporal buffer is removed before and after each spike.

The remaining volumes are considered safe low-motion regions. Fixed-size, non-overlapping chunks are then extracted from these safe regions and saved to an output CSV.

### Safe-Zone criteria

A volume is considered part of a safe zone if:

- Its FD is below the low-motion threshold.
- It is not within the buffer before a motion spike.
- It is not within the buffer after a motion spike.

This helps ensure that selected chunks are not only low-motion, but also temporally separated from nearby motion artefacts.

### Default Settings

| Parameter | Default | Description |
|---|---:|---|
| `chunk_size` | `20` | Number of volumes per extracted chunk |
| `thr_low` | `0.25 mm` | FD threshold used to identify motion spikes |
| `buf_before` | `5` | Volumes excluded before each spike |
| `buf_after` | `10` | Volumes excluded after each spike |
| `radius` | `50.0 mm` | Brain radius used to convert rotations into displacement |

and it saves the chunk in csv file as for the motion corrupted data. 


## CycleGAN Dataset Construction

The CycleGAN dataset was created using the script `build_cyclegan_dataset.py`. The goal of this script is to build an unpaired image-to-image translation dataset for fMRI motion artefact correction, where the two domains are:

* `A_corrupted`: motion-corrupted BOLD chunks
* `B_motion_free`: motion-free BOLD chunks

The input data consisted of three chunk-level CSV files: one CSV containing motion-corrupted chunks, one CSV containing motion-free chunks from the `video task`, and one CSV containing motion-free chunks from the `resting-state` task. Each row in these CSV files contains the subject/session/run information, the preprocessed BOLD file path, motion parameter file path, and the chunk start/end volume indices. The script uses the `preprocessed_bold_file` column for extracting chunks on preprocessed data. 

Motion-free chunks from the video and resting-state CSV files were combined into a single `B_motion_free` domain. Motion-corrupted chunks were used as the `A_corrupted` domain. Before extraction, the script checks that the preprocessed file exists and that the requested chunk indices are valid for the 4D BOLD time series.

To avoid data leakage, the dataset was split at the subject level using a `70/15/15` train/validation/test split. This means that each subject appears in exactly one split only, and all corrupted and motion-free chunks from that subject remain within the same split. This prevents chunks from the same subject or run appearing in both training and evaluation data.

Because the number of corrupted chunks is much larger than the number of motion-free chunks, balanced domain sampling was used. For training, all available motion-free chunks from training subjects were retained, while a balanced corrupted subset was generated for epoch 0. The full corrupted training pool was also saved separately so that new balanced corrupted subsets can be resampled during training if needed. For validation and test sets, all motion-free chunks were retained, and a fixed equal number of corrupted chunks was randomly sampled to create balanced evaluation sets.

The extracted chunks were saved as separate NIfTI files using the following CycleGAN directory convention:

```text
cyclegan_dataset/
├── train/
│   ├── A_corrupted/
│   └── B_motion_free/
├── val/
│   ├── A_corrupted/
│   └── B_motion_free/
├── test/
│   ├── A_corrupted/
│   └── B_motion_free/
└── metadata/
    ├── subject_split.csv
    ├── train_motion_free.csv
    ├── train_corrupted_all.csv
    ├── train_corrupted_balanced_epoch0.csv
    ├── val_motion_free.csv
    ├── val_corrupted_balanced.csv
    ├── test_motion_free.csv
    ├── test_corrupted_balanced.csv
    └── extraction_log.csv
```

The metadata files record the subject split, selected chunks for each domain and split, original source files, chunk start/end indices, motion statistics, and output chunk paths. The `extraction_log.csv` file records any missing files, invalid chunks, or skipped rows during dataset creation.

The script was run using:

```bash
python -u build_cycle_gans_dataset.py \
    --corrupted_csv "$CORRUPTED_CSV" \
    --motion_free_video_csv "$MOTION_FREE_VIDEO_CSV" \
    --motion_free_rest_csv "$MOTION_FREE_REST_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
```
or just run the slurm script to split the data into train, val, and test in chunks. 
```bash
sbatch run_build_cylegans_data_slurm_job.sh

tail -f logs/logs/build_cycleGAN_data_%j.out
```
