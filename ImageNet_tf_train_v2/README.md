### Install cv2
``` bash
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
apt install libgl1-mesa-glx -y
pip3 install opencv-python
```

### Install tqdm
``` bash
pip3 install tqdm
```

### ImageNet path
* IMAGENET_TRAIN_DIR = "/path/to/your/train/folder"
* IMAGENET_TEST_DIR = "/path/to/your/val/folder"
* Train Image path would be like: IMAGENET_TRAIN_DIR/n02268443/n02268443_2233.JPEG
* Test Image path would be like: IMAGENET_TEST_DIR/ILSVRC2012_val_00000293.JPEG