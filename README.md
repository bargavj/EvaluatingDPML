# Analyzing the Leaky Cauldron

The goal of this project is to evaluate the privacy leakage of differential private machine learning algorithms.

The code has been adapted from the [code base](https://github.com/csong27/membership-inference) of membership inference attack work by [Shokri et al.](https://ieeexplore.ieee.org/document/7958568)

Below we describe the procedure to run the experiments for the following projects:
* [Evaluating Differentially Private Machine Learning in Practice](#evaluating-differentially-private-machine-learning-in-practice)
* [Revisiting Membership Inference Under Realistic Assumptions](#revisiting-membership-inference-under-realistic-assumptions)


### Software Requirements

- Python 3.8 (https://www.anaconda.com/distribution/)
- Tensorflow (https://www.tensorflow.org/install)
- Tensorflow Privacy (https://github.com/tensorflow/privacy)


### Installation Instructions

Assuming the system has Ubuntu 18.04 OS. The easiest way to get Python 3.8 is to install [Anaconda 3](https://www.anaconda.com/distribution/) followed by installing the dependencies via pip. The following bash code installs the dependencies (including `scikit_learn`, `tensorflow>=2.4.0` and `tf-privacy`) in a virtual environment:

```
$ python3 -m venv env
$ source env/bin/activate
$ python3 -m pip install --upgrade pip
$ python3 -m pip install --no-cache-dir -r requirements.txt
```

Furthermore, to use cuda-compatible nvidia gpus, the following script should be executed (copied from [Tensorflow website](https://www.tensorflow.org/install/gpu)) to install cuda-toolkit-11 and cudnn-8 as required by tensorflow-gpu:

```
# Add NVIDIA package repositories
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
$ sudo apt-get update

$ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

$ sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
$ sudo apt-get update

$ wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
$ sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
$ sudo apt-get update

# Install development and runtime libraries (~4GB)
$ sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0

# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install TensorRT. Requires that libcudnn8 is installed above.
$ sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0
```


### Pre-processing data sets

Pre-processed CIFAR-100 data set has been provided in the `dataset/` folder. Purchase-100 data set can be downloaded from [Kaggle web site](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data). This can be pre-processed using the preprocess_purchase.py script provided in the repository. Alternatively, the files for Purchase-100 data set can be found [here](https://drive.google.com/open?id=1nDDr8OWRaliIrUZcZ-0I8sEB2WqAXdKZ).
For pre-processing other data sets, bound the L2 norm of each record to 1 and pickle the features and labels separately into `$DATASET`_feature.p and `$DATASET`_labels.p files in the `dataset/` folder (where `$DATASET` is a placeholder for the data set file name, e.g. for Purchase-100 data set, `$DATASET` will be `purchase_100`).


## Evaluating Differentially Private Machine Learning in Practice

To replicate the results from the paper [*Evaluating Differentially Private Machine Learning in Practice*](https://arxiv.org/abs/1902.08874), you would need to execute the `evaluating_dpml_run.sh` shell script, which runs the `evaluating_dpml.py` multiple times for different hyper-parameter settings and stores the results in the `results/$DATASET` folder. This is used for plotting the figures/tables in the paper. Note that the execution takes 3-5 days to complete on a single machine. For instance, for CIFAR-100 data set, run the following command:
```
$ ./evaluating_dpml_run.sh cifar_100
```
Note that the above script also installs all the required dependencies (in case not already installed), except cuda-toolkit and cudnn. For Purchase-100 data set, update the `target_l2_ratio` hyper-parameter as commented inside the script, and run:
```
$ ./evaluating_dpml_run.sh purchase_100
```

As mentioned above, the enitre script execution takes several days to finish as it requires running `evaluating_dpml.py` multiple times for all possible settings. This is required to generate all the tables/plots in the paper. However, we can also run `evaluating_dpml.py` for specific settings, as explained below.

When you are running the code on a data set for the first time, run:
```
$ python3 evaluating_dpml.py cifar_100 --save_data=1
```
This will split the data set into random subsets for training and testing of target, shadow and attack models.

To train a single non-private neural network model over CIFAR-100 data set, you can run: 
```
$ python3 evaluating_dpml.py cifar_100 --target_model='nn' --target_l2_ratio=1e-4
```
To train a single differentially private neural network model over CIFAR-100 data set using Rényi differential privacy with a privacy loss budget of 10, run:
```
$ python3 evaluating_dpml.py cifar_100 --target_model='nn' --target_l2_ratio=1e-4 --target_privacy='grad_pert' --target_dp='rdp' --target_epsilon=10
```


### Plotting the results from the paper 

Run `python3 evaluating_dpml_interpret_results.py $DATASET --model=$MODEL --l2_ratio=$LAMBDA` to obtain the plots and tabular results. For instance, to get the results for neural network model over CIFAR-100 data set, run:
```
$ python3 evaluating_dpml_interpret_results.py cifar_100 --model='nn' --l2_ratio=1e-4
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


## Revisiting Membership Inference Under Realistic Assumptions

To replicate the results of the paper [*Revisiting Membership Inference Under Realistic Assumptions*](https://arxiv.org/abs/2005.10881), use the same commands as above but replace `evaluating_dpml` with `improved_mi`. For instance, to run the batch file, run `./improved_mi_run.sh $DATASET` on terminal.

Run `python3 improved_mi_interpret_results.py $DATASET --l2_ratio=$LAMBDA` to obtain the plots and tabular results. Other command-line arguments are as follows: 
- `--plot` specifies the type of plot to be printed
    - 'acc' prints the accuracy loss comparison plot (default)
    - 'priv' prints the privacy leakage plots and table values
    - 'scatter' runs the Morgan attack and plots the scatter plot of loss and Merlin ratio
- `--gamma` specifies the gamma value to be used for the results: 1, 2 or 10
- `--alpha` specifies the alpha threshold to be used to get the corresponding attack threshold: between 0 and 1
- `--per_class_thresh` specifies whether to use per class threshold (1) or not (0 - default)
- `--fixed_thresh` specfies if fixed threshold of expected training loss is to be used when using per class threshold: set to 1 for using fixed threshold (0 - default)
- `--eps` specifies the epsilon value to be used when plotting 'priv' plots (None - default, i.e. no privacy)
