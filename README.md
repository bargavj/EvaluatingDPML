# Analyzing the Leaky Cauldron

The goal of this project is to evaluate the privacy leakage of differential private machine learning algorithms.

The code has been adapted from the [code base](https://github.com/csong27/membership-inference) of membership inference attack work by [Shokri et al.](https://ieeexplore.ieee.org/document/7958568)

Below we describe the setup and installation instructions. To run the experiments for the following projects, refer to their respective README files (hyperlinked):
* [Evaluating Differentially Private Machine Learning in Practice](evaluating_dpml/README.md) (`evaluating_dpml\`)
* [Revisiting Membership Inference Under Realistic Assumptions](improved_mi/README.md) (`improved_mi\`)


### Software Requirements

- [Python 3.8](https://www.anaconda.com/distribution/)
- [Tensorflow](https://www.tensorflow.org/install) : To use Tensorflow with GPU, cuda-toolkit-11 and cudnn-8 are also [required](https://www.tensorflow.org/install/gpu).
- [Tensorflow Privacy](https://github.com/tensorflow/privacy)


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
