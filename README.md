## Motion correction with CyleGANS
preprocessing details to be addd. 


### Building chunked low-motion and motion corrupted data 
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
