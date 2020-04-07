#!/bin/bash

echo "Starting script"
if [[ $# -eq 0 ]] ; then # if called with no arguments
    echo "Usage: bash $0 [-s] <DATASET NAME> <RUN>"
    echo "	-s: saves new sample of dataset to data/ directory"
    exit 1
fi

if [ $1 == '-s' ]; then
	DATASET=$2
	SAVE_DATA=true
	RUN=$3
else
	DATASET=$1
	SAVE_DATA=false
	RUN=$2
fi

ATTACK_PY=attack.py

echo "Loading modules"
source /etc/profile.d/modules.sh
module load anaconda3

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
	python $ATTACK_PY $DATASET --use_cpu=0 --save_data=1 --target_test_train_ratio=10
fi

echo "Beginning experiment"
for GAMMA in 1 2 10
	do
	python $ATTACK_PY $DATASET --use_cpu=0 --target_test_train_ratio=$GAMMA --target_epochs=100 --target_learning_rate=0.005 --target_privacy='no_privacy' --run=$RUN
	for EPSILON in 0.1 0.5 1.0 5.0 10.0 50.0 100.0 500.0 1000.0
	do
		python $ATTACK_PY $DATASET --use_cpu=0 --target_test_train_ratio=$GAMMA --target_epochs=100 --target_learning_rate=0.005 --target_privacy='grad_pert' --target_dp='rdp' --target_epsilon=$EPSILON --run=$RUN
	done
	for EPSILON in 0.1 0.5 1.0 5.0 10.0 50.0 100.0
	do
		python $ATTACK_PY $DATASET --use_cpu=0 --target_test_train_ratio=$GAMMA --target_epochs=100 --target_learning_rate=0.005 --target_privacy='grad_pert' --target_dp='gdp' --target_epsilon=$EPSILON --run=$RUN
	done
done
echo done

