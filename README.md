# Analyzing the Leaky Cauldron

The goal of this project is to evaluate the privacy leakage of differential private machine learning algorithms.

The code has been adapted from the [code base](https://github.com/csong27/membership-inference) of membership inference attack work by [Shokri et al.](https://ieeexplore.ieee.org/document/7958568)

Below we describe the setup and installation instructions. To run the experiments for the following projects, refer to their respective README files (hyperlinked):
* [Evaluating Differentially Private Machine Learning in Practice](evaluating_dpml/README.md) (`evaluating_dpml\`)
* [Revisiting Membership Inference Under Realistic Assumptions](improved_mi/README.md) (`improved_mi\`)
* [Are Attribute Inference Attacks Just Imputation?](improved_ai/README.md) (`improved_ai\`)


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


### Obtaining the Data Sets

Data sets can be obtained using the `preprocess_dataset.py` script provided in the `extra/` folder. The script requires raw files for the respective data sets which can be found online using the following links:

- **Purchase-100X**: The source file `transactions.csv` can be downloaded from https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data and should be saved in the `dataset/` folder.
- **Compas**: The source file `cox-violent-parsed_filt.csv` file can be downloaded from https://www.kaggle.com/danofer/compass and should be saved in the `dataset/` folder.
- **Census19**: The source files can be downloaded from https://www2.census.gov/programs-surveys/acs/data/pums/2019/1-Year/ and should be saved in the `dataset/census/` folder. Alternatively, the source files can be obtained by running the `crawl_census_data.py` script in the `extra/` folder:
>> `$ python3 crawl_census_data.py`
- **Texas-100X**: `PUDF_base1q2006_tab.txt`, `PUDF_base2q2006_tab.txt`, `PUDF_base3q2006_tab.txt` and `PUDF_base4q2006_tab.txt` files can be downloaded from https://www.dshs.texas.gov/THCIC/Hospitals/Download.shtm and should be saved in the `dataset/texas_100_v2/` folder.
- **Location**: The source file `bangkok_location` can be downloaded from https://github.com/privacytrustlab/datasets and should be saved in the `dataset/` folder. 

Once the source files for the respective data set are obtained, `preprocess_dataset.py` script would be able to generate the processed data set files, which are in the form of two pickle files: `$DATASET`_feature.p and `$DATASET`_labels.p (where `$DATASET` is a placeholder for the data set file name). For Purchase-100X, `$DATASET = purchase_100`. For Texas-100X, `$DATASET = texas_100_v2`. For Compas, `$DATASET = compas`. For Location, `$DATASET = location`. For Census19, `$DATASET = census`.
```
$ python3 preprocess_dataset.py $DATASET --preprocess=1
```

For pre-processing other data sets, bound the L2 norm of each record to 1 and pickle the features and labels separately into `$DATASET`_feature.p and `$DATASET`_labels.p files in the `dataset/` folder (where `$DATASET` is a placeholder for the data set file name, e.g. for Purchase-100 data set, `$DATASET` will be `purchase_100`).
