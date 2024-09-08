# Dementia Prediction Codebase

This repository contains code to run machine learning experiments aimed at predicting dementia using various data modalities. The code is organized into several folders, each corresponding to a different data modality:

- `cognitive_tests`
- `neuroimaging`
- `proteomics`

## Folder Structure

Each modality folder contains the following scripts:

1. **build_ml_datasets.py**:
   - Combines the modality dataset with demographics and dementia data.
   - Removes patients with dementia at or before the modality time point.
   - Encodes categorical and/or ordinal variables.
   - Saves labels and region indices for cross-validation.
   
   Reference: 
   ```python:proteomics/build_ml_datasets.py
   startLine: 1
   endLine: 50
   ```

2. **ml_experiments.py**: 
   - Runs machine learning experiments with various specifications.
   - Options include different experiment types (age only, all demographics, modality variables only, modality variables and demographics).
   - Supports different loss functions (1-AUROC, 1-F3 score, 1-average precision).
   - Allows for an age cutoff and time budget for AutoML.
   - Saves true labels and predicted probabilities, and selects decision thresholds based on the loss function used during training.
   
   Reference: 
   ```python:proteomics/ml_experiments.py
   startLine: 41
   endLine: 259
   ```

3. **sh_ml_experiments.sh**: 
   - Script called by `sbatch` to specify parameters for `ml_experiments.py`.
   - Current design of `age_cutoff` needs to be fixed.
   
   Reference: 
   ```shell:proteomics/sh_ml_experiments.sh
   startLine: 1
   endLine: 34
   ```

4. **loop_ml.sh**: 
   - Loops over experiment types and loss functions to call `sbatch` on `sh_ml_experiments.sh`.
   
   Reference: 
   ```README
   startLine: 14
   endLine: 14
   ```

## Additional Scripts

- **feature_selection_experiments.py**: 
  - Runs feature selection experiments.
  - Supports different experiment types and metrics.
  
  Reference: 
  ```python:proteomics/feature_selection_experiments.py
  startLine: 47
  endLine: 259
  ```

- **rfb.py**: 
  - Runs random feature baselines pipeline.
  - Loads or creates datasets, processes data, and runs AutoML.
  
  Reference: 
  ```python:neurips_2024/rfb.py
  startLine: 1
  endLine: 300
  ```

- **concatenate_results.py**: 
  - Concatenates results from multiple files into a single DataFrame.
  
  Reference: 
  ```python:all_outcomes/concatenate_results.py
  startLine: 1
  endLine: 21
  ```

## Environment Setup

To set up the environment, use the `requirements.txt` file located in the `ukb_func` folder. This file can be used to create a conda environment:
