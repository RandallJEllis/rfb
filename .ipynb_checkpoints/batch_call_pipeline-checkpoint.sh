#!/bin/bash

numbers="256, 392, 521, 138, 17, 406, 280, 544, 545, 546, 295, 297, 554, 555, 556, 45, 557, 54, 311, 312, 438, 440, 187, 443, 444, 446, 319, 64, 321, 448, 70, 327, 72, 328, 458, 459, 590, 85, 88, 344, 349, 94, 481, 355, 483, 229, 359, 104, 233, 106, 234, 237, 112, 368, 381, 506, 507, 253, 255"

# Set the Internal Field Separator (IFS) to comma
IFS=','

for i in $numbers; do
#for i in {2..608}; do
   sbatch --export=MYVAR=$i sh_pipeline.sh
done
