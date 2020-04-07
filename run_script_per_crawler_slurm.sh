#!/bin/bash

#SBATCH --job-name="DiPrivML"
#SBATCH --ntasks=100

echo "Starting script"
if [[ $# -eq 0 ]] ; then # if called with no arguments
    echo "Usage: sbatch $0 <DATASET NAME>"
    exit 1
fi

DATASET=$1
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

conda create -n tf tensorflow
source activate tf
conda install scikit-learn

pip install dm-tree
pip install matplotlib
pip install git+git://github.com/tensorflow/privacy@master

echo "Filling data/ directory"
python $ATTACK_PY $DATASET --use_cpu=1 --save_data=1 --target_test_train_ratio=10

echo "Beginning experiment"
# parallel will run the quoted command, replacing each {n} with each value on
# the n-th line starting with ":::" exactly as if it were a nested for-loop.
# The start-times, durations, and commands of tasks are stored in joblog.txt.
parallel -j 20 "srun --exclusive -N1 -n1 python $ATTACK_PY $DATASET \
    --target_privacy='no_privacy' \
    --use_cpu=1 \
    --target_epochs=100 \
    --target_learning_rate=0.005 \
    --target_test_train_ratio={1} \
    --run={2}" ::: 1 2 10 ::: 1 2 3 4 5
parallel -j $SLURM_NTASKS --joblog joblog.txt --ungroup \
    "srun --exclusive -N1 -n1 python $ATTACK_PY $DATASET \
    --target_privacy='grad_pert' \
    --use_cpu=1 \
    --target_epochs=100 \
    --target_learning_rate=0.005 \
    --target_test_train_ratio={1} \
    --target_dp={2} \
    --target_epsilon={3} \
    --run={4}" ::: 1 2 10 ::: 'gdp' 'rdp' ::: 0.1 0.5 1.0 5.0 10.0 50.0 100.0 ::: 1 2 3 4 5
echo done
