import os
# input("pid: " + str(os.getpid()) +", press enter after attached")
import numpy as np
import tensorflow as tf
import argparse
print("tf verison: " + tf.__version__)
# input("pid: " + str(os.getpid()) +", press enter after set breakpoints")
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Start with XLA disabled.
tf.debugging.set_log_device_placement(True)
import config as c
from utils.data_utils import load_tfr_dataset

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--case", default = "res50_tf_npu", type = str)
args = parser.parse_args()

MODEL_FILE = "resnet.json"
MODEL_DATA_FILE = "resnet.h5"

# DUMP_DIR = "res50_tf_floordiv" # "npu"
DUMP_DIR = args.case
MODEL_NAME = "offical_v1" # "tiny_v1_for_cifar"
DATA_DIR = "./fake_data/resnet50"
IMAGENET_TFR_DIR = "/dataset/imagenet_gideon/tf_records"

FULL_TEST=False
ENABLE_MULTI_DEVICE = False

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())

def load_cifar_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32') / 256
  x_test = x_test.astype('float32') / 256

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))

def load_imagenet_data():
  train_ds = load_tfr_dataset(IMAGENET_TFR_DIR, "train/train*")
  test_ds = load_tfr_dataset(IMAGENET_TFR_DIR, "val/validation*")
  train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
  test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True)
  return (train_ds, test_ds)
  # train_data = train_ds.__iter__()
  # test_data = test_ds.__iter__()
  # return (train_data, test_data)

# fake data
def load_fake_data(train_size, test_size, height, width, channel, num_classes):
  rng = np.random.default_rng(12345)
  x_train = rng.random((train_size, height, width, channel))
  x_test = rng.random((test_size, height, width, channel))

  y_train = rng.integers(low=0, high=1000, size=(train_size,1))
  y_test = rng.integers(low=0, high=1000, size=(test_size,1))
  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
  return ((x_train, y_train), (x_test, y_test))

def load_fake_data_file(train_size, test_size, height, width, channel, num_classes):
  x_train = np.loadtxt(DATA_DIR + '/' + "x_train.txt")
  y_train = np.loadtxt(DATA_DIR + '/' + "y_train.txt")
  x_test = np.loadtxt(DATA_DIR + '/' + "x_test.txt")
  y_test = np.loadtxt(DATA_DIR + '/' + "y_test.txt")
  x_train = np.reshape(x_train, (-1, height, width, channel))
  x_test = np.reshape(x_test, (-1, height, width, channel))
  y_train = np.reshape(y_train, (-1, num_classes))
  y_test = np.reshape(y_test, (-1, num_classes))
  return ((x_train, y_train), (x_test, y_test))

"""
this resnetv1 demo which come from:
https://blog.csdn.net/my_name_is_learn/article/details/109640715
"""

def generate_tiny_model():
  input_layer = tf.keras.layers.Input((224, 224, 3))
  conv1 = Conv_BN_Relu(64, (3, 3), 1, input_layer)
  x = residual_a_or_b(conv1, 64, 'b')
  x = residual_a_or_b(x, 64, 'a')

  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(1000)(x)
  y = tf.keras.layers.Softmax(axis=-1)(x)
  model = tf.keras.models.Model([input_layer], [y])
  return model

def generate_v2_model():
  model = tf.keras.applications.resnet_v2.ResNet50V2()
  return model

def generate_v1_model():
  model = tf.keras.applications.resnet50.ResNet50()
  return model

def compile_model(model):
  opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
  # opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,)
                # run_eagerly=True,
                # metrics=['accuracy'])
  return model

def train_model(model, x_train, y_train, x_test,
  y_test, epochs=1, batch_size=1):
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
    validation_data=(x_test, y_test), shuffle=False)

def warmup(model, x_train, y_train, x_test, y_test):
  # Warm up the JIT, we do not wish to measure the compilation time.
  initial_weights = model.get_weights()
  train_model(model, x_train, y_train, x_test, y_test, epochs=1)
  model.set_weights(initial_weights)

def get_model(model_list, model_name):
  for model in model_list:
    if (model['name'] == model_name):
      return model

model_list = [
    {'name': "offical_v2",
     'model': 'generate_v2_model()',
     'data_set': 'load_fake_data(TRAIN_SIZE, 1, 224, 224, 3, 1000)',
    },
    {'name': "offical_v1",
     'model': 'generate_v1_model()',
     'data_set': 'load_imagenet_data()',
    },
]

model_info = get_model(model_list, MODEL_NAME)
print(model_info)

BATCH_SIZE = 2
if FULL_TEST:
  EPOCHS = 100
  # TRAIN_SIZE = 100
else:
  EPOCHS = 50
  TRAIN_SIZE = BATCH_SIZE

train_data, test_data = eval(model_info['data_set'])

def load_model():
  if os.path.exists(MODEL_FILE):
    json_string = open(MODEL_FILE, 'r').read() 
    model = tf.keras.models.model_from_json(json_string)
    model.load_weights(MODEL_DATA_FILE)
  else:
    model = eval(model_info['model'])
  return model

if ENABLE_MULTI_DEVICE:
  with strategy.scope():
    model = load_model()
else:
  model = load_model()

model = compile_model(model)
model.summary()
model.fit(train_data, batch_size=BATCH_SIZE, epochs=1,
  validation_data=test_data, shuffle=False)

model.save_weights(DUMP_DIR + "/after/" + MODEL_DATA_FILE)

if not os.path.exists(MODEL_FILE):
  json_string = model.to_json()
  open(MODEL_FILE, 'w').write(json_string) 
  model.save_weights(MODEL_DATA_FILE)
  print("RRR : save model.")

# outputs = [layer.output for layer in model.layers][1:]
# # all layer outputs except first (input) layer
# functor = tf.keras.backend.function([model.input], outputs)
# layer_outs = functor([x_train])
# dump_tensors(layer_outs, "output", outputs)

# print(model.predict(x_test))

print("RRR : job finish.")
