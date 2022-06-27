# Install bazel
# tf2.5.0
# _TF_MIN_BAZEL_VERSION = '3.7.2'
# _TF_MAX_BAZEL_VERSION = '3.99.0'
apt install -y apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

apt update
apt install -y bazel-3.7.2
ln -s /usr/bin/bazel-3.7.2 /usr/bin/bazel
bazel --version