#!/bin/bash

# numbers="38"

# Set the Internal Field Separator (IFS) to comma
#IFS=','

# for i in $numbers; do
for i in {0..0}; do
   sbatch --export=MYVAR=$i sh_pipeline_hipfracture_dementia_neurips.sh
done
