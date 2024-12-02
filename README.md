# Dementia Prediction Codebase

This repository contains code to run machine learning experiments aimed at predicting dementia using various data modalities. There are three core scripts:

1. **ml_experiments.py**: 
   - Runs machine learning experiments with various specifications.
   - Options include different experiment types (age only, all demographics, modality variables only, modality variables and demographics, and feature selection of conditions including the modality).
   - Supports different loss functions (1-AUROC, 1-F3 score, 1-average precision).
   - Allows for an age cutoff and time budget for AutoML.
   - Saves true labels and predicted probabilities, and selects decision thresholds based on the loss function used during training.

2. **sh_ml_experiments.sh**: 
   - Script called by `sbatch` to specify parameters for `ml_experiments.py`.
   - Current design of `age_cutoff` needs to be fixed.

3. **loop_ml.sh**: 
   - Loops over experiment types and loss functions to call `sbatch` on `sh_ml_experiments.sh`.

There is a corresponding set of three scripts for time to event experiments:

1. **timetoevent_experiments.py**
2. **sh_timetoevent_experiments.sh**
3. **loop_timetoevent.sh**

The code is organized into folders, each corresponding to a different data modality:

- `cognitive_tests`
- `neuroimaging`
- `proteomics`

## Folder Structure

Each modality folder contains the following scripts:
- **build_ml_datasets.py**:
   - Combines the modality dataset with demographics and dementia data.
   - Removes patients with dementia at or before the modality time point.
   - Encodes categorical and/or ordinal variables.
   - Saves labels and region indices for cross-validation.

## Environment Setup

To set up the environment, use the `requirements.txt` file located in the `ukb_func` folder. This file can be used to create a conda environment:
