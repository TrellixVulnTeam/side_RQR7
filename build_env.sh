# build_env
Username = 'gideon.wu'
sudo apt install python3-pip
sudo apt install vim

# ------------------------------
# install cudnn
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/10.2_20201106/cudnn-10.2-linux-x64-v8.0.5.39.tgz
tar zxvf udnn-10.2-linux-x64-v8.0.5.39.tgz

sudo cp cuda/include/cudnn.h    /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn*    /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h   /usr/local/cuda/lib64/libcudnn*

# ------------------------------
# install cuda (network)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# install cuda (local)
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

# ------------------------------
# Install Docker
#Update the apt package index and install packages to allow apt to use a repository over HTTPS:
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker’s official GPG key:
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker.io  
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

# Step 1

dpkg -l | grep -i docker
# To identify what installed package you have:

# Step 2
sudo apt-get purge -y docker-engine docker docker.io docker-ce docker-ce-cli
sudo apt-get autoremove -y --purge docker-engine docker docker.io docker-ce  

# Step 3
sudo rm -rf /var/lib/docker /etc/docker
sudo rm /etc/apparmor.d/docker
sudo groupdel docker
sudo rm -rf /var/run/docker*



# ------------------------------
# Nvidia container
# 1、添加源
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# 2、安装并重启
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
# sudo apt-get install -y nvidia-container-runtime
sudo systemctl restart docker
# 3、测试
docker run --name test1 -it --rm --gpus all ubuntu nvidia-smi

# ------------------------------

# Install Sublime On Debian/Ubuntu
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install sublime-text
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
alias tf25='sudo docker run --gpus all -it --rm --shm-size="1g" --name=gideon_tf-2.5.0 -w=/workspace -v /home/gideon:/workspace -v /home/d/gideon_dataset/:/dataset tensorflow/tensorflow:2.5.0-gpu /bin/bash'

alias torch17='sudo docker run --gpus all -it --rm --shm-size="1g" --name=gideon_pytorch -w=/workspace -v /home/gideon:/workspace -v /home/d/gideon_dataset/:/dataset pytorch/pytorch:latest /bin/bash'

#--------------------------------
# pruning package
pip3 instaall tensorboard prefetch-generator nni


#--------------------------------
#VM nvidia driver
cd /home/cad
./NVIDIA-*.run -no-x-check -no-nouveau-check -no-opengl-files 
.run path /home/cad/Download
