sudo adduser gideon
sudo adduser gideon sudo

# Check GPU
lspci | grep -i NVIDIA

# Install GPU driver
wget https://tw.download.nvidia.com/tesla/470.82.01/NVIDIA-Linux-x86_64-470.82.01.run

sudo apt-get purge nvidia*

sudo apt update
sudo apt install gcc make cmake dkms build-essential lib32ncurses5 lib32z1

sudo bash NVIDIA-Linux-x86_64-470.82.01.run

sudo reboot

# ------------------------------
# Install CUDA 11.5
wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda_11.5.0_495.29.05_linux.run
sudo sh cuda_11.5.0_495.29.05_linux.run

# vim ~/.bashrc
sudo sh -c "echo 'export PATH=\$PATH:/usr/local/cuda-11.5/bin' >> /etc/profile"
sudo sh -c "echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-11.5/lib64' >> /etc/profile"
export PATH=$PATH:/usr/local/cuda-11.5/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.5/lib64

# ------------------------------
# Install cudnn
wget https://developer.nvidia.com/compute/cudnn/secure/8.3.1/local_installers/11.5/

sudo dpkg -i cudnn-local-repo-ubuntu1804-8.3.1.22_1.0-1_amd64.deb
sudo apt-key add /var/cudnn-local-repo-ubuntu1804-8.3.1.22/7fa2af80.pub

sudo apt-get update
sudo apt-get install libcudnn8=8.3.1.22-1+cuda11.5
sudo apt-get install libcudnn8-dev=8.3.1.22-1+cuda11.5
sudo apt-get install libcudnn8-samples=8.3.1.22-1+cuda11.5

# Verify installation
sudo apt-get install libfreeimage3 libfreeimage-dev
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN

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
sudo docker version
# add your user to the docker group. To do so, simply type the following:
sudo groupadd docker
sudo usermod -aG docker $USER

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

# systemctl start docker
# systemctl enable docker

# On Linux, you can also run the following command to activate the changes to groups:
newgrp docker 
sudo systemctl restart docker
