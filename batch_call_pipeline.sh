#!/bin/bash

numbers="229, 70, 521, 106, 88"

# Set the Internal Field Separator (IFS) to comma
IFS=','

for i in $numbers; do
#for i in {2..608}; do
   sbatch --export=MYVAR=$i sh_pipeline.sh
done
