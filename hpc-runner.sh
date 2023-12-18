#!/bin/bash -l

# Parse arguments
N_STEP=$1
ALPHA=$2

echo "[Debug] N_STEP: $N_STEP, ALPHA: $ALPHA"
conda activate pyncn310
echo "[Debug] Env activated"
cd $SCRATCH/Files/GitRepos/nSARSA/
python solution.py -ne "5000" -n "$N_STEP" -a "$ALPHA"
echo "[Debug] Deactivating env"
conda deactivate
