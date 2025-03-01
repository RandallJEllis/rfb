# CLAUDE.md - Quick Reference Guide

## Build/Run Commands
- Python: `python file.py [args]`
- R: `Rscript file.R` or run within RStudio
- ML experiments: `python ml_experiments.py --modality "$modality" --experiment "$experiment" --metric "$metric" --model "$model" --region_index "$region_index" --age_cutoff "$age_cutoff" --predict_alzheimers_only "$predict_alzheimers_only"`
- Shell scripts: Most are SLURM job submission wrappers (e.g., `sh_ml_experiments.sh`, `loop_ml.sh`)

## Code Style Guidelines
### Python
- Imports: Standard library first, third-party packages second, local modules last
- Formatting: 4-space indentation, snake_case for functions/variables
- Error handling: Minimal explicit error handling in most files
- Uses pandas, numpy, scikit-learn heavily; data stored in parquet/CSV formats

### R
- Libraries: Imported at top with `library()`
- Formatting: 2-space indentation, uses tidyverse piping (`%>%`)
- Working directory: Set with `setwd(dirname(rstudioapi::getActiveDocumentContext()$path))`
- Statistical focus: Joint models, time-varying covariates, survival analysis

## Project Structure
- Analysis organized by task/domain (proteomics, neuroimaging, A4, etc.)
- Common utilities in `ukb_func/`
- Region-based analysis with fold indexing (0-9)
- Results often saved in nested directory structures