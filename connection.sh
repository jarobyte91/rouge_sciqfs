#!/bin/sh
salloc --time=3:0:0 --ntasks=1 --cpus-per-task=4 --mem=64G --account=rrg-emilios srun /home/jarobyte/envs/malnis/bin/notebook.sh
