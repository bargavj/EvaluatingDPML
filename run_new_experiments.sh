#!/bin/bash

#SBATCH --job-name="DiPrivML"
#SBATCH --ntasks=100

echo "Starting script"
if [[ $# -eq 0 ]] ; then # if called with no arguments
    echo "Usage: sbatch $0 [-s] <DATASET NAME> <TEST TO TRAIN RATIO>"
    echo "	-s: saves new sample of dataset to data/ directory"
    exit 1
fi

if [ $1 == '-s' ]; then
	DATASET=$2
	SAVE_DATA=true
    GAMMA=$3
else
	DATASET=$1
	SAVE_DATA=false
    GAMMA=$2
fi

ATTACK_PY=attack.py

echo "Loading modules"
source /etc/profile.d/modules.sh
module load anaconda3
module load parallel

# Make sure conda environment has dependencies
echo "Creating conda environment"
conda create -n tf-gpu tensorflow-gpu
source activate tf-gpu
conda install scikit-learn

pip install dm-tree
pip install matplotlib
pip install git+git://github.com/tensorflow/privacy@master

if [ "$SAVE_DATA" = true ]; then
	echo "Filling data/ directory"
	python $ATTACK_PY $DATASET --save_data=1 --target_test_train_ratio=$GAMMA
fi

echo "Beginning experiment"
# parallel will run the quoted command, replacing each {n} with each value on
# the n-th line starting with ":::" exactly as if it were a nested for-loop.
# The start-times, durations, and commands of tasks are stored in joblog.txt.
parallel -j 20 "python $ATTACK_PY $DATASET \
    --use_cpu=0 \
    --target_privacy='no_privacy' \
    --target_model='nn' \
    --target_l2_ratio=1e-8 \
    --target_learning_rate=0.005 \
    --run={1}" ::: 1 2 3 4 5
parallel -j 20 --joblog joblog.txt --ungroup \
    "python $ATTACK_PY $DATASET \
    --use_cpu=0 \
    --target_privacy='grad_pert' \
    --target_model='nn' \
    --target_l2_ratio=1e-8 \
    --target_learning_rate=0.005 \
    --target_dp='rdp' \
    --target_epsilon={1} \
    --run={2}" ::: 0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0 100.0 500.0 1000.0 ::: 1 2 3 4 5
parallel -j 20 --joblog joblog.txt --ungroup \
    "python $ATTACK_PY $DATASET \
    --use_cpu=0 \
    --target_privacy='grad_pert' \
    --target_model='nn' \
    --target_l2_ratio=1e-8 \
    --target_learning_rate=0.005 \
    --target_dp='gdp' \
    --target_epsilon={1} \
    --run={2}" ::: 0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0 100.0 ::: 1 2 3 4 5
echo done
