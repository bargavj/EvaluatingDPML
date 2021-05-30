# Evaluating Differentially Private Machine Learning in Practice

Please refer to [main README file](../README.md) for instructions on setup and installation.

To replicate the results from the paper [*Evaluating Differentially Private Machine Learning in Practice*](https://arxiv.org/abs/1902.08874), you would need to execute the `run_experiments.sh` shell script, which runs the `main.py` multiple times for different hyper-parameter settings and stores the results in the `results/$DATASET` folder. This is used for plotting the figures/tables in the paper. Note that the execution takes 3-5 days to complete on a single machine. For instance, for CIFAR-100 data set, run the following command:
```
$ ./run_experiments.sh cifar_100
```
Note that the above script also installs all the required dependencies (in case not already installed), except cuda-toolkit and cudnn. For Purchase-100 data set, update the `target_l2_ratio` hyper-parameter as commented inside the script, and run:
```
$ ./run_experiments.sh purchase_100
```

As mentioned above, the enitre script execution takes several days to finish as it requires running `main.py` multiple times for all possible settings. This is required to generate all the tables/plots in the paper. However, we can also run `main.py` for specific settings, as explained below.

When you are running the code on a data set for the first time, run:
```
$ python3 main.py cifar_100 --save_data=1
```
This will split the data set into random subsets for training and testing of target, shadow and attack models.

To train a single non-private neural network model over CIFAR-100 data set, you can run: 
```
$ python3 main.py cifar_100 --target_model='nn' --target_l2_ratio=1e-4
```
To train a single differentially private neural network model over CIFAR-100 data set using Rényi differential privacy with a privacy loss budget of 10, run:
```
$ python3 main.py cifar_100 --target_model='nn' --target_l2_ratio=1e-4 --target_privacy='grad_pert' --target_dp='rdp' --target_epsilon=10
```


### Plotting the results from the paper 

Run `python3 interpret_results.py $DATASET --model=$MODEL --l2_ratio=$LAMBDA` to obtain the plots and tabular results. For instance, to get the results for neural network model over CIFAR-100 data set, run:
```
$ python3 interpret_results.py cifar_100 --model='nn' --l2_ratio=1e-4
```

Other command-line arguments are as follows: 
- `--function` prints the plots if set to 1 (default), or gives the membership revelation results at fixed FPR if set to 2, or gives the membership revelation results at fixed threshold if set to 3.
- `--plot` specifies the type of plot to be printed
    - 'acc' prints the accuracy loss comparison plot (default)
    - 'shokri_mi' prints the privacy leakage due to Shokri et al. membership inference attack
    - 'yeom_mi' prints the privacy leakage due to Yeom et al. membership inference attack
    - 'yeom_ai' prints the privacy leakage due to Yeom et al. attribute inference attack
- `--silent` specifies if the plot values are to be displayed (0) or not (1 - default)
- `--fpr_threshold` sets the False Positive Rate threshold (refer the paper)
- `--venn` plots the venn diagram of members identified by MI attack across two runs when set to 1, otherwise it does not plot when set to 0 (default). This functionality works only when `--function=3`

