import os
import numpy as np
import tensorflow as tf

# Generate Training Dataset (x_train, x_test, y_train, y_test)

DUMP_DIR = "./fake_data"

# fake data
def gen_fake_data(train_size, test_size, height, width, channel, num_classes):
    rng = np.random.default_rng(12345)
    x_train = rng.random((train_size, height, width, channel))
    x_test = rng.random((test_size, height, width, channel))

    y_train = rng.integers(low=0, high=1000, size=(train_size,1))
    y_test = rng.integers(low=0, high=1000, size=(test_size,1))
    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    print(x_train.shape)
    print(y_train.shape)
    return ((x_train, y_train), (x_test, y_test))

def dump_tensors(tensors, prefix, tensors_name=None):
    if not os.path.isdir(DUMP_DIR): os.makedirs(DUMP_DIR)
    if not os.path.isdir(DUMP_DIR + "/" + prefix):
        os.makedirs(DUMP_DIR + "/" + prefix)
    ((x_train, y_train), (x_test, y_test)) = tensors
    # print(train[0])
    # print(train[1])
    # print(len(x_train))
    x_train_name = DUMP_DIR + "/" + prefix + "/" + "x_train.txt"
    x_test_name = DUMP_DIR + "/" + prefix + "/" + "x_test.txt"
    y_train_name = DUMP_DIR + "/" + prefix + "/" + "y_train.txt"
    y_test_name = DUMP_DIR + "/" + prefix + "/" + "y_test.txt"
    np.savetxt(x_train_name, x_train.flatten(), fmt='%.8f')
    np.savetxt(x_test_name, x_test.flatten(), fmt='%.8f')
    np.savetxt(y_train_name, y_train.flatten(), fmt='%.5d')
    np.savetxt(y_test_name, y_test.flatten(), fmt='%.5d')
        # np.savetxt(x_train_name, x_train.flatten(), fmt='%.8f')

def main():
    
    TRAIN_SIZE = 5
    TEST_SIZE = 1
    HEIGHT = 224
    WIDTH = 224
    CHANNEL = 3
    NUM_CLASSES = 1000
    data = gen_fake_data(
        TRAIN_SIZE, TEST_SIZE, HEIGHT,
        WIDTH, CHANNEL, NUM_CLASSES)
    dump_tensors(data, 'resnet50')

if __name__ == '__main__':
    main()
    print("{} Job Finish !!! {}".format('='*5, '='*5))