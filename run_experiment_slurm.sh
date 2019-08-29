#!/bin/bash

#SBATCH --job-name="DiPrivML"
#SBATCH --ntasks=100

echo "Starting script"
if [[ $# -eq 0 ]] ; then # if called with no arguments
    echo "Usage: sbatch $0 [-s] <DATASET NAME>"
    echo "	-s: saves new sample of dataset to data/ directory"
    exit 1
fi

if [ $1 == '-s' ]; then
	DATASET=$2
	SAVE_DATA=true
else
	DATASET=$1
	SAVE_DATA=false
fi

ATTACK_PY=attack.py

echo "Loading modules"
source /etc/profile.d/modules.sh
module load anaconda3
module load parallel

# Make sure conda environment has dependencies
echo "Creating conda environment"
conda env remove -y -n slurm || true
conda create -n slurm || true
source activate slurm
conda install -y pip
conda install -y python
while read r; do
	pip install $r
done <requirements.txt

# flags for compatibility between tensorflow and theano
export KMP_DUPLICATE_LIB_OK=TRUE
export THEANO_FLAGS=device=cpu

if [ "$SAVE_DATA" = true ]; then
	echo "Filling data/ directory"
	python $ATTACK_PY $DATASET --save_data=1
fi

echo "Beginning experiment"
# parallel will run the quoted command, replacing each {n} with each value on
# the n-th line starting with ":::" exactly as if it were a nested for-loop.
# The start-times, durations, and commands of tasks are stored in joblog.txt.
parallel -j $SLURM_NTASKS --joblog joblog.txt --ungroup \
    "srun --exclusive -N1 -n1 python $ATTACK_PY $DATASET \
    --target_privacy='grad_pert' \
    --target_model={1} \
    --target_l2_ratio={2} \
    --target_dp={3} \
    --target_epsilon={4} \
    --run={5}" ::: 'softmax' 'nn' :::+ '1e-5' '1e-4' ::: 'dp' 'rdp' ::: 0.5 1.0 5.0 10.0 50.0 ::: 1 2 3 4 5
parallel -j 20 "srun --exclusive -N1 -n1 python $ATTACK_PY $DATASET \
    --target_privacy='no_privacy' \
    --target_model={1} \
    --target_l2_ratio={2} \
    --run={3}" ::: 'softmax' 'nn' :::+ '1e-5' '1e-4' ::: 1 2 3 4 5
echo done
