#!/bin/bash

# We have 22 specific visit values to process
# Arrays are zero-indexed, so we use 0-21
total_visits=21  # 22 visits, indexed 0-21

# Submit array job
sbatch --job-name="PACC_Analysis" \
       --output="stdouterr/PACC_Analysis_%A_%a.out" \
       --error="stdouterr/PACC_Analysis_%A_%a.err" \
       --array=0-$total_visits \
       pacc.sh