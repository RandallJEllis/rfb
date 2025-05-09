#!/bin/bash

nums="97,98,99,100"

# Submit the job array with specific indices
sbatch --array=$nums --export=ALL sh_pipeline.sh

# 1-607