# Revisiting Membership Inference Under Realistic Assumptions

Please refer to [main README file](../README.md) for instructions on setup and installation.

To replicate the results of the paper [*Are Attribute Inference Attacks Just Imputation?*](https://arxiv.org/abs/2005.10881), you would need to execute the `run_experiments.sh` shell script, which runs the `main.py` multiple times for different hyper-parameter settings and stores the results in the `results/$DATASET` folder. This is used for plotting the figures/tables in the paper. Note that the execution may take multiple days to complete on a single machine. For instance, for Census19 data set, run the following command:
```
$ ./run_experiments.sh -s census
```
Note that the above script also installs all the required dependencies (in case not already installed), except cuda-toolkit and cudnn. For Texas-100X data set, update the hyper-parameters as commented inside the script, and run:
```
$ ./run_experiments.sh -s texas_100_v2
```


### Plotting the results from the paper 

Run `interpret_results.py` to obtain the plots and tabular results. For instance, to get the results for Census19 data set, run:
```
$ python3 interpret_results.py census --skew_attribute=0 --attribute=1 --skew_outcome=3 --sensitive_outcome=3 --adv_knowledge='med' --sample_size=50
```    

Other command-line arguments are as follows: 
- `--skew_attribute`: Attribute on which to skew the non-iid data sampling 0 (population, default), 1 or 2 -- for Census 1: Income and 2: Race, and for Texas 1: Charges and 2: Ethnicity
- `--attribute`: Target sensitive attribute. For Census19: 1 (Race), and for Texas-100X: 2 (Ethnicity)
- `--skew_outcome`: In case skew_attribute = 1, which outcome the distribution is skewed upon -- For Census Race: 0 (White, default), 1 (Black) or 3 (Asian), and for Texas Ethnicity: 0 (Hispanic, default) or 1 (Not Hispanic)
- `--sensitive_outcome`: For Census Race: 0 (White, default), 1 (Black) or 3 (Asian), and for Texas Ethnicity: 0 (Hispanic, default) or 1 (Not Hispanic)
- `--eps` specifies the epsilon value to be used for private model (default = None), use eps=1 for private model results
- `--adv_knowledge`: Distribution knowledge of adversary: knows skewed distribution (low or low2), knows training distribution (med) or knows training data set (high)
- `--sample_size`: How many records the adversary knows: 50, 500, 5000 or 50000
- `--banished_records`: Whether to get results from model trained after removing vulnerable records
