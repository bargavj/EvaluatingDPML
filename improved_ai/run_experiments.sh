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
module load python3.6.2

# Make sure environment has dependencies
echo "Creating environment"
python3 -m venv ../env
source ../env/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r ../requirements.txt

if [ $DATASET == 'census' ]; then
    L2_RATIO=1e-6
    SKEW_OUT=3
    SENS_OUT=3
    SENS_ATTR=1
else
    L2_RATIO=1e-3
    SKEW_OUT=0
    SENS_OUT=0
    SENS_ATTR=2
fi

echo "Filling data/ directory"
python3 $CODE $DATASET --use_cpu=1 --save_data=1 --skew_attribute=0 --skew_outcome=$SKEW_OUT --sensitive_outcome=$SENS_OUT --target_test_train_ratio=0.5 --target_data_size=50000

echo "Beginning experiment"
# Experiment Setting : 50K training set size, 25K test set size, 10K candidate size
# For Texas-100-v2, set --target_epochs=50, --target_learning_rate=0.001, --target_l2_ratio=1e-3, --target_batch_size=500, --target_data_size=50000, --candidate_size=10000, --targrt_test_train_ratio=0.5
# For Census, set --target_epochs=50, --target_learning_rate=0.001, --target_l2_ratio=1e-6, --target_batch_size=500, --target_data_size=50000, --candidate_size=10000, --targrt_test_train_ratio=0.5
for RUN in 1 2 3 4 5
do
    python3 $CODE $DATASET \
    --use_cpu=1 \
    --skew_attribute=0 \
    --skip_corr=1 \
    --skew_outcome=$SKEW_OUT \
    --sensitive_outcome=$SENS_OUT \
    --target_test_train_ratio=0.5 \
    --target_data_size=50000 \
    --candidate_size=10000 \
    --target_model='nn' \
    --target_epochs=50 \
    --target_l2_ratio=$L2_RATIO \
    --target_learning_rate=0.001 \
    --target_batch_size=500 \
    --target_clipping_threshold=4 \
    --attribute=$SENS_ATTR \
    --run=$RUN
done
# Differential private model training
for RUN in 1 2 3 4 5
do
    python3 $CODE $DATASET \
    --use_cpu=1 \
    --skew_attribute=0 \
    --skip_corr=1 \
    --skew_outcome=$SKEW_OUT \
    --sensitive_outcome=$SENS_OUT \
    --target_test_train_ratio=0.5 \
    --target_data_size=50000 \
    --candidate_size=10000 \
    --target_model='nn' \
    --target_epochs=50 \
    --target_l2_ratio=$L2_RATIO \
    --target_learning_rate=0.001 \
    --target_batch_size=500 \
    --target_clipping_threshold=4 \
    --target_privacy='grad_pert' \
    --attribute=$SENS_ATTR \
    --target_epsilon=1 \
    --run=$RUN
done
# Model training after removing sensitive records
# Note: run the below script only after running interpret_results.py
# to generate banished_records.p file.
for RUN in 1 2 3 4 5
do
    python3 $CODE $DATASET \
    --use_cpu=1 \
    --skew_attribute=0 \
    --banished_records=1 \
    --skip_corr=1 \
    --skew_outcome=$SKEW_OUT \
    --sensitive_outcome=$SENS_OUT \
    --target_test_train_ratio=0.5 \
    --target_data_size=50000 \
    --candidate_size=10000 \
    --target_model='nn' \
    --target_epochs=50 \
    --target_l2_ratio=$L2_RATIO \
    --target_learning_rate=0.001 \
    --target_batch_size=500 \
    --target_clipping_threshold=4 \
    --attribute=$SENS_ATTR \
    --run=$RUN
done
echo done
