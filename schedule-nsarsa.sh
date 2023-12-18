#!/bin/bash -l

TIMESTAMP=$(date +%y%m%d_%H%M%S)
FOLDER_SPEC="nsarsa"
GITREPO="nSARSA"

# Create directory structure to store logs
mkdir -p $SCRATCH/Logs/$FOLDER_SPEC/
mkdir -p $SCRATCH/Logs/$FOLDER_SPEC/$TIMESTAMP/


ALPHA_SPAN=20
N_STEP_SPAN=9

for (( n = 1; n < N_STEP_SPAN; n++ )); do
    for (( a = 0; a < ALPHA_SPAN; a++ )); do

        N_STEP=$((2**n))
        ALPHA=$(echo "scale=5; 1.25^(-$a)" | bc)

        LOG_OUTPUT_DIR=$SCRATCH/Logs/$FOLDER_SPEC/$TIMESTAMP/n$N_STEP-a$ALPHA-output.out
        LOG_ERROR_DIR=$SCRATCH/Logs/$FOLDER_SPEC/$TIMESTAMP/n$N_STEP-a$ALPHA-error.err
        sbatch --job-name="nsarsa" --account="plgsano4-cpu" --partition="plgrid" --output="$LOG_OUTPUT_DIR" --error="$LOG_ERROR_DIR" --time="4:00:00" --nodes="1" --ntasks-per-node="1" --cpus-per-task="1" --mem-per-cpu="1GB" $SCRATCH/Files/GitRepos/$GITREPO/hpc-runner.sh $N_STEP $ALPHA
    done
done
