# Build Docker under the docker folder
sudo docker build -t tim-vx-xla . --no-cache --rm

# Install bash
bash bazel_3.7.2_install.sh

# Build Tensorflow under the tensorflow folder
bash tf_build.sh

# Enviroment var
source env_export.sh