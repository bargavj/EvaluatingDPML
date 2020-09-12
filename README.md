# Analysing the Leaky Cauldron

The goal of this project is to evaluate the privacy leakage of differential private machine learning algorithms.

The code has been adapted from the [code base](https://github.com/csong27/membership-inference) of membership inference attack work by [Shokri et al.](https://ieeexplore.ieee.org/document/7958568)

Below we describe the procedure to run the experiments for the following projects:
* [Evaluating Differentially Private Machine Learning in Practice](#evaluating-differentially-private-machine-learning-in-practice)
* [Revisiting Membership Inference under Realistic Assumptions](#revisiting-membership-inference-under-realistic-assumptions)


### Requirements

- Python 2.7 or higher (https://www.anaconda.com/distribution/)
- Tensorflow (https://www.tensorflow.org/install)
- Tensorflow Privacy (https://github.com/tensorflow/privacy)

### Pre-processing data sets

Pre-processed CIFAR-100 data set has been provided in the `dataset/` folder. Purchase-100 data set can be downloaded from [Kaggle web site](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data). This can be pre-processed using the preprocess_purchase.py scipt provided in the repository. Alternatively, the files for Purchase-100 data set can be found [here](https://drive.google.com/open?id=1nDDr8OWRaliIrUZcZ-0I8sEB2WqAXdKZ).
For pre-processing other data sets, bound the L2 norm of each record to 1 and pickle the features and labels separately into `$dataset`_feature.p and `$dataset`_labels.p files in the `dataset/` folder (where `$dataset` is a placeholder for the data set file name, e.g. for Purchase-100 data set, `$dataset` will be purchase_100).


## Evaluating Differentially Private Machine Learning in Practice

Follow the instructions below to replicate the results from the paper [*Evaluating Differentially Private Machine Learning in Practice*](https://arxiv.org/abs/1902.08874).

### Training the non-private baseline models for CIFAR

When you are running the code on a data set for the first time, run `python evaluating_dpml.py $dataset --save_data=1` on terminal. This will split the data set into random subsets for training and testing of target, shadow and attack models.

Run `python evaluating_dpml.py $dataset --target_model=$model --target_l2_ratio=$lambda` on terminal.

For training optimal non-private baseline neural network on CIFAR-100 data set, we set `$dataset`='cifar_100', `$model`='nn' and `$lambda`=1e-4. For logsitic regression model, we set `$dataset`='cifar_100', `$model`='softmax' and `$lambda`=1e-5.

For training optimal non-private baseline neural network on Purchase-100 data set, we set `$dataset`='purchase_100', `$model`='nn' and `$lambda`=1e-8. For logsitic regression model, we set `$dataset`='cifar_100', `$model`='softmax' and `$lambda`=1e-5.

### Training the differential private models

Run `python evaluating_dpml.py $dataset --target_model=$model --target_l2_ratio=$lambda --target_privacy='grad_pert' --target_dp=$dp --target_epsilon=$epsilon` on terminal. Where `$dp` can be set to 'dp' for naive composition, 'adv_cmp' for advanced composition, 'zcdp' for zero concentrated DP and 'rdp' for Renyi DP. `$epsilon` controls the privacy budget parameter. Refer to __main__ block of evaluating_dpml.py for other command-line arguments.

### Plotting the results from the paper 

Update the `$lambda` variables accordingly and run `./evaluating_dpml_run.sh $dataset` on terminal. Results will be stored in `results/$dataset` folder.

Run `evaluating_dpml_interpret_results.py $dataset --model=$model --l2_ratio=$lambda` to obtain the plots and tabular results. Other command-line arguments are as follows: 
- `--function` prints the plots if set to 1 (default), or gives the membership revelation results at fixed FPR if set to 2, or gives the membership revelation results at fixed threshold if set to 3.
- `--plot` specifies the type of plot to be printed
    - 'acc' prints the accuracy loss comparison plot (default)
    - 'shokri_mi' prints the privacy leakage due to Shokri et al. membership inference attack
    - 'yeom_mi' prints the privacy leakage due to Yeom et al. membership inference attack
    - 'yeom_ai' prints the privacy leakage due to Yeom et al. attribute inference attack
- `--silent` specifies if the plot values are to be displayed (0) or not (1 - default)
- `--fpr_threshold` sets the False Positive Rate threshold (refer the paper)
- `--venn` plots the venn diagram of members identified by MI attack across two runs when set to 1, otherwise it does not plot when set to 0 (default). This functionality works only when `--function=3`


## Revisiting Membership Inference under Realistic Assumptions

To replicate the results of the paper [*Revisiting Membership Inference under Realistic Assumptions*](https://arxiv.org/abs/2005.10881), use the same commands as above but replace `evaluating_dpml` with `improved_mi`. For instance, to run the batch file, run `./improved_mi_run.sh $dataset` on terminal.

Run `improved_mi_interpret_results.py $dataset --l2_ratio=$lambda` to obtain the plots and tabular results. Other command-line arguments are as follows: 
- `--plot` specifies the type of plot to be printed
    - 'acc' prints the accuracy loss comparison plot (default)
    - 'priv' prints the privacy leakage plots and table values
    - 'scatter' runs the Morgan attack and plots the scatter plot of loss and Merlin ratio
- `--gamma` specifies the gamma value to be used for the results: 1, 2 or 10
- `--alpha` specifies the alpha threshold to be used to get the corresponding attack threshold: between 0 and 1
- `--per_class_thresh` specifies whether to use per class threshold (1) or not (0 - default)
- `--fixed_thresh` specfies if fixed threshold of expected training loss is to be used when using per class threshold: set to 1 for using fixed threshold (0 - default)
- `--eps` specifies the epsilon value to be used when plotting 'priv' plots (None - default, i.e. no privacy)
