#!/bin/bash

echo "Starting script"
if [[ $# -eq 0 ]] ; then # if called with no arguments
    echo "Usage: bash $0 <DATASET NAME>"
    exit 1
fi

DATASET=$1
CODE=main.py

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

echo "Filling data/ directory"
# For Texas-100 and CIFAR-100, set --target_test_train_ratio=2 
# For RCV1, set --target_clipping_threshold=1
python $CODE $DATASET --save_data=1 --target_clipping_threshold=4 --target_test_train_ratio=10

echo "Beginning experiment"
# For Purchase-100, set GAMMA in 0.1 0.5 1 2 10, --target_clipping_threshold=4, --target_epochs=100, --target_learning_rate=0.005, --target_l2_ratio=1e-8
# For Texas-100, set GAMMA in 0.1 0.5 1 2, --target_clipping_threshold=4, --target_epochs=30, --target_learning_rate=0.005, --target_l2_ratio=1e-8
# For RCV1, set GAMMA in 0.1 0.5 1 2 10, --target_clipping_threshold=1, --target_epochs=80, --target_learning_rate=0.003, --target_l2_ratio=1e-8
# For CIFAR-100, set GAMMA in 0.1 0.5 1 2, --target_clipping_threshold=4, --target_epochs=100, --target_learning_rate=0.001, --target_l2_ratio=1e-4
for GAMMA in 0.1 0.5 1 2 10
do
    for RUN in 1 2 3 4 5
    do
        python $CODE $DATASET --target_test_train_ratio=$GAMMA --target_model='nn' --target_l2_ratio=1e-8 --target_learning_rate=0.005 --target_clipping_threshold=4 --target_privacy='no_privacy' --run=$RUN
        for EPSILON in 0.1 1.0 10.0 100.0
        do
            python $CODE $DATASET --target_test_train_ratio=$GAMMA --target_model='nn' --target_l2_ratio=1e-8 --target_learning_rate=0.005 --target_clipping_threshold=4 --target_privacy='grad_pert' --target_dp='rdp' --target_epsilon=$EPSILON --run=$RUN
            python $CODE $DATASET --target_test_train_ratio=$GAMMA --target_model='nn' --target_l2_ratio=1e-8 --target_learning_rate=0.005 --target_clipping_threshold=4 --target_privacy='grad_pert' --target_dp='gdp' --target_epsilon=$EPSILON --run=$RUN
        done
    done
done
echo done