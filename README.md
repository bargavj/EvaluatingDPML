# Evaluating Differentially Private Machine Learning in Practice

This repository contains code used in the paper (https://arxiv.org/abs/1902.08874). The code evaluates the utility and privacy leakage of some differential private machine learning algorithms.

This code has been adapted from the code base (https://github.com/csong27/membership-inference) of membership inference attack work by Shokri et al. (https://ieeexplore.ieee.org/document/7958568).


### Pre-Processing Data Sets

Pre-processed CIFAR-100 data set has been provided in the `dataset/` folder. Purchase-100 data set can be downloaded from Kaggle web site (https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data). This can be pre-processed using the preprocess_purchase.py scipt provided in the repository.
For pre-processing other data sets, bound the L2 norm of each record to 1 and pickle the features and labels separately into `$dataset`_feature.p and `$dataset`_labels.p files in the `dataset/` folder.


### Training the Non-Private Baseline Models for CIFAR

When you are running the code on a data set for the first time, run `python attack_tf.py $dataset --save_data=1` on terminal. This will split the data set into random subsets for training and testing of target, shadow and attack models.

Run `python attack_tf.py $dataset --target_model=$model --target_l2_ratio=$lambda` on terminal.

For training optimal non-private baseline neural network on CIFAR-100 data set, we set `$dataset`='cifar_100', `$model`='nn' and `$lambda`=1e-4. For logsitic regression model, we set `$dataset`='cifar_100', `$model`='softmax' and `$lambda`=1e-5.

For training optimal non-private baseline neural network on Purchase-100 data set, we set `$dataset`='purchase_100', `$model`='nn' and `$lambda`=1e-8. For logsitic regression model, we set `$dataset`='cifar_100', `$model`='softmax' and `$lambda`=1e-5.


### Training the Differential Private Models

Run `python attack_tf.py $dataset --target_model=$model --target_l2_ratio=$lambda --target_privacy='grad_pert' --target_dp=$dp --target_epsilon=$epsilon` on terminal. Where `$dp` can be set to 'dp' for naive composition, 'adv_cmp' for advanced composition, 'zcdp' for zero concentrated DP and 'rdp' for Renyi DP. `$epsilon` controls the privacy budget parameter. Refer to __main__ block of attack_tf.py for other command-line arguments.


### Simulating the Experiments from the Paper 

Update the `$dataset`, `$model` and `$lambda` variables accordingly and run `./run_experiment.sh` on terminal. Results will be stored in `results/$dataset` folder.

Run `interpret_results.py $dataset --model=$model --l2_ratio=$lambda` to obtain the plots and tabular results. Other command-line arguments are as follows: 
- `--function` prints the plots if set to 1 (default), or gives the membership revelation results if set to 2.
- `--plot` specifies the type of plot to be printed
    - 'acc' prints the accuracy loss comparison plot (default)
    - 'attack' prints the privacy leakage due to Shokri et al. membership inference attack
    - 'mem' prints the privacy leakage due to Yu et al. membership inference attack
    - 'attr' prints the privacy leakage due to Yu et al. attribute inference attack
- `--silent` specifies if the plot values are to be displayed (0) or not (1 - default)
- `--fpr_threshold` sets the False Positive Rate threshold (refer the paper)
