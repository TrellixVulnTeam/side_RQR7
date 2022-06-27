ln -s /usr/bin/python3 /usr/bin/python
bazel clean --expunge
./configure
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package && \
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tmp/tensorflow_pkg && \
pip3 install ./tmp/tensorflow_pkg/*