# When Relaxations Go Bad: "Differentially-Private" Machine Learning
This repository contains code used in the paper (https://arxiv.org/abs/1902.08874). The code evaluates the utility and privacy leakage of some differential private machine learning algorithms.

This code has been adapted from the code base (https://github.com/csong27/membership-inference) of membership inference attack work by Shokri et al. (https://ieeexplore.ieee.org/document/7958568).

### Pre-Processing Data Sets
Pre-processed CIFAR-100 data set has been provided in the 'dataset/' folder. For pre-processing other data sets, bound the L2 norm of each record to 1 and pickle the features and labels separately into $dataset_feature.p and $dataset_labels.p files in the 'dataset/' folder.

### Training the Non-Private Baseline Models for CIFAR

Run 'python attack.py $dataset --target_model=$model --target_l2_ratio=$lambda' on terminal.

For training optimal non-private baseline neural network on CIFAR-100 data set, we set $dataset='cifar_100', $model='nn' and $lambda=1e-4. For logsitic regression model, we set $dataset='cifar_100', $model='softmax' and $lambda=1e-5.

For training optimal non-private baseline neural network on Purchase-100 data set, we set $dataset='purchase_100', $model='nn' and $lambda=1e-8. For logsitic regression model, we set $dataset='cifar_100', $model='softmax' and $lambda=1e-5.

### Training the Differential Private Models

Update the $dataset, $model and $lambda variables accordingly and run './run_experiment.sh' on terminal. Results will be stored in 'results/' folder.

Update the interpret_results.py file accordingly and run it to obtain the final results.
