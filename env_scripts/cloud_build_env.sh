# build_env
Username = 'gideon.wu'
sudo apt-get update
sudo apt-get -y install python3
sudo apt install -y python3-pip
sudo apt install vim

sudo shutdown -h now

# ------------------------------
# install Nvidia Driver
# ------------------------------

## 清除既有driver
sudo apt-get purge nvidia*
## 加入GPU ppa
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt upgrade
## 列出支援的GPU driver版本
ubuntu-drivers list
## nvidia-driver-460版本安裝
############# Work ##############
sudo apt-get install -y nvidia-driver-460
reboot
## check
nvidia-smi

# ------------------------------
# Uninstall Nvidia Drivers
# ------------------------------
sudo apt clean
sudo apt update
# To remove CUDA Toolkit:
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
 "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" 
# To remove NVIDIA Drivers:
sudo apt-get --purge remove "*nvidia*"
# To clean up the uninstall:
sudo apt-get autoremove

# ------------------------------
# Install CUDA 11.2 for ubuntu 20.04 x86_64
# ------------------------------
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev

## Local
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
############# Work ##############
sudo sh cuda_11.2.0_460.27.04_linux.run
# 會有對話框，要選continue跟accept
# 如果已經裝完driver 則選toolkit安裝即可

## Network
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
sudo apt install -y cuda

## deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.2-460.32.03-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
sudo apt install -y cuda

# 安裝完CUDA 添加
############# Work ##############
# vi ~/.bashrc
vi /etc/profile
export PATH=$PATH:/usr/local/cuda-11.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/extras/CUPTI/lib64
# source ~/.bashrc
source /etc/profile

## check
nvcc -V

# ------------------------------
# Install cuDNN
# ------------------------------
## Download cudnn 8.1
#  https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz
############# Work ##############
tar zxvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
sudo cp cuda/include/cudnn.h    /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn*    /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h   /usr/local/cuda/lib64/libcudnn*

# ------------------------------
# Install Docker
# ------------------------------
#Update the apt package index and install packages to allow apt to use a repository over HTTPS:
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker’s official GPG key:
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl start docker
systemctl enable docker
docker version

# add your user to the docker group. To do so, simply type the following:
sudo groupadd docker
sudo usermod -aG docker $USER

# On Linux, you can also run the following command to activate the changes to groups:
newgrp docker
sudo systemctl restart docker

# ------------------------------
# Uninstall Docker
# ------------------------------
# Step 1
dpkg -l | grep -i docker
# To identify what installed package you have:

# Step 2
sudo apt-get purge -y docker-engine docker docker.io docker-ce docker-ce-cli
sudo apt-get autoremove -y --purge docker-engine docker docker.io docker-ce 
sudo apt-get remove docker docker-engine docker.io containerd runc

# Step 3
sudo rm -rf /var/lib/docker /etc/docker
sudo rm /etc/apparmor.d/docker
sudo groupdel docker
sudo rm -rf /var/run/docker*


# ------------------------------
# Nvidia container
# ------------------------------
# docker run --gpu
# 1、添加源
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 2、安装并重启
# sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-docker2
# sudo apt-get install -y nvidia-container-runtime
sudo systemctl restart docker
# 3、测试
docker run --name test1 -it --rm --gpus all ubuntu nvidia-smi


# ------------------------------
# Build TF on GPU docker
# ------------------------------
docker pull nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
docker run --gpus all -it --rm --name=gpu_env -w=/workspace -v /home/ubuntu:/workspace nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 /bin/bash

apt update
apt install -y python3-dev python3-pip
apt install -y git wget libgoogle-glog-dev
apt install -y libboost-all-dev --fix-missing
pip3 install pip numpy wheel packaging requests opt_einsum keras_preprocessing 

ln -s /usr/bin/python3 /usr/bin/python 

# Install Bazel
cd tensorflow_folder
./configure
# CUDA: Y
# https://developer.nvidia.com/cuda-gpus#compute
# cuda compute capability: ex M60 is 5.2
# Do you want to use clang as CUDA compiler? [y/N]: N
bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tmp/tensorflow_pkg

# ------------------------------
# Install Sublime On Debian/Ubuntu
# ------------------------------
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install sublime-text
# ------------------------------

# ------------------------------
# TensorFlow - GPU:
# python
from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())

# PyTorch - GPU:
# python
import torch
id = torch.cuda.current_device()
print(id)
print(torch.cuda.get_device_name(id))

#--------------------------------
alias tf27='sudo docker run --gpus all -it --rm --shm-size="1g" --name=gideon_tf-2.7.0 -w=/workspace -v /home/gideon:/workspace -v /home/d/gideon_dataset/:/dataset tensorflow/tensorflow:2.7.0-gpu /bin/bash'

alias torch17='sudo docker run --gpus all -it --rm --shm-size="1g" --name=gideon_pytorch -w=/workspace -v /home/gideon:/workspace -v /home/d/gideon_dataset/:/dataset pytorch/pytorch:latest /bin/bash'

#--------------------------------
# pruning package
pip3 install tensorboard prefetch-generator nni


#--------------------------------
#VM nvidia driver
cd /home/cad
./NVIDIA-*.run -no-x-check -no-nouveau-check -no-opengl-files 
.run path /home/cad/Download


docker run --gpus all -it --rm --shm-size="1g" --name=gideon_tf-2.7.0 -w=/workspace -v /home/ubuntu:/workspace  tensorflow/tensorflow:2.7.0-gpu /bin/bash