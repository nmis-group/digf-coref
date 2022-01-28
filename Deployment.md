# Deployment on EDGE device

## Edge device

Device information()


Edge device information - the Stereolabs [ZED Box Xavier NX](https://www.stereolabs.com/zed-box/) comes with two NVIDIA® Jetson™ Module options. 


## System info

Reference [ZED Box Xavier NX](https://www.stereolabs.com/zed-box/) hardware information online.


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

Create the environment
```sh
conda create -n deep python==3.8.

python -m ipykernel install --user --name deep --display-name "DL"
```

Add the dependency package for deep learning IDE. Normally, the nessesory package for deep learning depends on the chosing library. Following list of package is an exmple.


```sh

conda install -c conda-forge pillow

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install -c conda-forge pandas

conda install -c conda-forge opencv

conda install -c conda-forge libopencv

conda install -c conda-forge py-opencv

conda install -c conda-forge matplotlib

conda install -c conda-forge scikit-learn

conda install -c conda-forge albumentations

conda install -c conda-forge pytorch-lighting

conda install -c conda-forge tensorboard
```

B. Alternative way to implement the conda environment is to directly use the pre-defined [environment yaml](/environment.yml) file with the correct deep learning and computer vision package such as pytorch, opencv, etc. 
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



