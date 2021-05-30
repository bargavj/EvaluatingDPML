# Revisiting Membership Inference Under Realistic Assumptions

Please refer to [main README file](../README.md) for instructions on setup and installation.

To replicate the results of the paper [*Revisiting Membership Inference Under Realistic Assumptions*](https://arxiv.org/abs/2005.10881), you would need to execute the `run_experiments.sh` shell script, which runs the `main.py` multiple times for different hyper-parameter settings and stores the results in the `results/$DATASET` folder. This is used for plotting the figures/tables in the paper. Note that the execution may take multiple days to complete on a single machine. For instance, for Purchase-100 data set, run the following command:
```
$ ./run_experiments.sh purchase_100
```
Note that the above script also installs all the required dependencies (in case not already installed), except cuda-toolkit and cudnn. For CIFAR-100 data set, update the hyper-parameters as commented inside the script, and run:
```
$ ./run_experiments.sh cifar_100
```


### Plotting the results from the paper 

Run `python3 interpret_results.py $DATASET --l2_ratio=$LAMBDA` to obtain the plots and tabular results. For instance, to get the results for neural network model over CIFAR-100 data set, run:
```
$ python3 interpret_results.py cifar_100 --model='nn' --l2_ratio=1e-4
```

Other command-line arguments are as follows: 
- `--plot` specifies the type of plot to be printed
    - 'acc' prints the accuracy loss comparison plot (default)
    - 'priv' prints the privacy leakage plots and table values
    - 'scatter' runs the Morgan attack and plots the scatter plot of loss and Merlin ratio
- `--gamma` specifies the gamma value to be used for the results: 1, 2 or 10
- `--alpha` specifies the alpha threshold to be used to get the corresponding attack threshold: between 0 and 1
- `--per_class_thresh` specifies whether to use per class threshold (1) or not (0 - default)
- `--fixed_thresh` specfies if fixed threshold of expected training loss is to be used when using per class threshold: set to 1 for using fixed threshold (0 - default)
- `--eps` specifies the epsilon value to be used when plotting 'priv' plots (None - default, i.e. no privacy)
- `--mem` used in scatter plot to specifify which points are to be plotted: 'm' to plot only members, 'nm' to plot only non-members ('all' - default, i.e. plot both members and non-members)
