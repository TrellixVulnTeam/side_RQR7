apt update
apt install -y python3-dev python3-pip
apt install -y git wget libgoogle-glog-dev
apt-get install -y libboost-all-dev --fix-missing
pip install pip numpy wheel packaging requests opt_einsum keras_preprocessing 

ln -s /usr/bin/python3 /usr/bin/python
bazel clean --expunge
./configure
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package && \
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tmp/tensorflow_pkg && \
pip3 install ./tmp/tensorflow_pkg/*