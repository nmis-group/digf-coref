# Deployment on EDGE device

## Edge device

Device information()


Edge device information - the Stereolabs [ZED Box Xavier NX](https://www.stereolabs.com/zed-box/) comes with two NVIDIA® Jetson™ Module options. 


## System info

Reference [ZED Box Xavier NX](https://www.stereolabs.com/zed-box/) hardware information online.
a

## Deployment 

### 1 Deployment Conda 

Download and install the latest conda for ARM64 (https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-aarch64.sh).

[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

A. Step by step deploy the developement environment by command line:

Implement the deep learning environment.

Add channels to the conda
```sh
conda config --add channels https://repo.anaconda.com/pkgs/main

conda config --add channels https://conda.anaconda.org/anaconda

conda config --add channels conda-forge

```

Create the environment:
```sh
conda create -n <environment name> python==3.6.

python -m ipykernel install --user --name <env name> --display-name "DL"
```

Replace "\<environment name>" with your environemt name.

### PyTorch on Jetson platform

Following the [instruction](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) to install the PyTorch on Jetson.

**Note: Please be aware the correct python version for proper PyTorch**
**Note: numpy 1.19.5 will cause [Illegal instruction (core dumped)](https://forums.developer.nvidia.com/t/illegal-instruction-core-dumped/165488) error when import torch in python interactive cli**


### Add deep learning libraries

Add the dependency package for deep learning IDE. Normally, the nessesory package for deep learning depends on the chosing library. 

Recommended libraries list:

- pillow
- pandas
- opencv
- libopencv
- py-opency
- matplotlib
- scikit-learn
- pytorch
- pytorch-lighting
- trochvison
- torchaudio
- cudatoolkit
- albumentations
- tensorboard

B. Alternative way to implement the conda environment is to directly use the pre-defined [environment yaml](../environment.yml) file with the correct deep learning and computer vision package such as pytorch, opencv, etc. 
Get the environment.yml file from github repository,

run the following command from terminal
```sh
conda create -n <environment name>

conda activiate <environment name>

conda env update -f environment.yml    
```


### 2 Deployment ZED SDK

https://www.stereolabs.com/docs/get-started-with-zed-box/

*If you know the jetpack version, skip this step and install the ZED SDK by type command from bash.*

Check nvidia Jetson version

```sh
apt-cache policy nvidia-jetpack
```

Here, the version is 

```sh
nvidia-jetpack:
  Installed: 4.6-b199
  Candidate: 4.6-b199
  Version table:
 *** 4.6-b199 500
        500 https://repo.download.nvidia.com/jetson/t194 r32.6/main arm64 Packages
        100 /var/lib/dpkg/status
     4.6-b197 500
        500 https://repo.download.nvidia.com/jetson/t194 r32.6/main arm64 Packages

```


Then, go to stereolabs.com/developers/release, click on the “SDK Downloads” tab and scroll down to ﬁnd the corresponding SDK for the Jetpack version of your NVIDIA® Jetson™ system.

Then we should download ZED SDK for [Jetpack 4.6](https://download.stereolabs.com/zedsdk/3.6/jp46/jetsons)

```sh
chmod +x ZED_SDK_<PLATFORM>_<VERSION>.run
    
./ZED_SDK_<PLATFORM>_<VERSION>.run
```



### Hit:
  When use the ZED camera at the first time, the internet connection should be ensured in case of the downloading some module of ZED.



