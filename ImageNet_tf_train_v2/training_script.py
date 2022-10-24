import os
# input("pid: " + str(os.getpid()) +", press enter after attached")
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
from models.resnet_model import resnet50 as resnet50V1_5
import config as c
from utils.data_utils import load_dataset
from utils.eval_utils import cross_entropy_batch, correct_num_batch, l2_loss


print("tf verison: " + tf.__version__)
print("pid: ", os.getpid())
time.sleep(3)
# input("pid: " + str(os.getpid()) +", press enter after set breakpoints")
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Start with XLA disabled.
# tf.debugging.set_log_device_placement(True)

MODEL_FILE = "resnet.json"
MODEL_DATA_FILE = "resnet.h5"

DUMP_DIR = "res50V1_5_tf_npu" # "npu"
MODEL_NAME = "resnet50_v1.5_for_imagenet" # "resnet50_v1.5_for_imagenet"
IMAGENET_TRAIN_DIR = "/dataset/imagenet_gideon/train"
IMAGENET_TEST_DIR = "/dataset/imagenet_gideon/val"

# Dataset config
TRAIN_LIST_PATH = 'label/train_label.txt'
TEST_LIST_PATH = 'label/validation_label.txt'

FULL_TEST=False
ENABLE_MULTI_DEVICE = False
BATCH_SIZE = 64

model_list = [
    {'name': "offical_v2",
     'model': 'generate_v2_model()',
     'data_set': 'load_fake_data(TRAIN_SIZE, 1, 224, 224, 3, 1000)',
    },
    {'name': "offical_v1",
     'model': 'generate_v1_model()',
     'data_set': 'load_fake_data(TRAIN_SIZE, 1, 224, 224, 3, 1000)',
    },
    {'name': "resnet50_v1.5",
     'model': 'resnet50V1_5(1000)',
     'data_set': 'load_fake_data(TRAIN_SIZE, 1, 224, 224, 3, 1000)',
    },
    {'name': "resnet50_v1.5_for_imagenet",
     'model': 'resnet50V1_5(1000)',
     'data_set': 'load_imagenet_data()',
    },
]


def load_cifar_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32') / 256
  x_test = x_test.astype('float32') / 256

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))

def load_imagenet_data():
  train_ds = load_dataset(TRAIN_LIST_PATH, IMAGENET_TRAIN_DIR)
  test_ds = load_dataset(TEST_LIST_PATH, IMAGENET_TEST_DIR)
  train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
  test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True)
  train_data = train_ds.__iter__()
  test_data = test_ds.__iter__()
  return (train_data, test_data)

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
                optimizer=opt,
                # run_eagerly=True,
                metrics=['accuracy'])
  return model

def get_model(model_list, model_name):
  for model in model_list:
    if (model['name'] == model_name):
      return model

def load_model():
  if os.path.exists(MODEL_FILE):
    json_string = open(MODEL_FILE, 'r').read() 
    model = tf.keras.models.model_from_json(json_string)
    model.load_weights(MODEL_DATA_FILE)
  else:
    model = eval(model_info['model'])
  return model

def dump_tensors(tensors, prefix, tensors_name=None):
  if not os.path.isdir(DUMP_DIR): os.makedirs(DUMP_DIR)
  if not os.path.isdir(DUMP_DIR + "/" + prefix):
    os.makedirs(DUMP_DIR + "/" + prefix)
  if (tensors_name == None):
    ts_zip = zip(tensors, tensors)
  else:
    ts_zip = zip(tensors, tensors_name)
  for tensor in ts_zip:
    file_name = DUMP_DIR + "/" + prefix + "/" + \
      tensor[1].name.replace("/", "_").replace(":", "_") + ".txt"
    print(tensor[1].name, tensor[1].shape, " saved in: ", file_name)
    if hasattr(tensor[0], "numpy"):
      num = tensor[0].numpy()
    else:
      num = tensor[0]
    np.savetxt(file_name, num.flatten(), fmt='%.8f')

def get_weight_grad(model, inputs, outputs):
  with tf.GradientTape() as tape:
    pred = model(inputs)
    loss = model.compiled_loss(tf.convert_to_tensor(outputs), pred, None,
                                   regularization_losses=model.losses)
  grad = tape.gradient(loss, model.trainable_weights)
  return grad

@tf.function
def train_step(model, images, labels, optimizer):
  with tf.GradientTape() as tape:
    prediction = model(images, training=True)
    ce = cross_entropy_batch(labels, prediction)
    l2 = l2_loss(model)
    loss = ce + l2
    gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return prediction, l2

def train(model, data_iterator):

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
  sum_ce = 0
  sum_correct_num = 0
  iterations_per_epoch = int(c.train_num / BATCH_SIZE)
  train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
  for i in tqdm(range(iterations_per_epoch)):
    images, labels = data_iterator.next()
    prediction, loss = train_step(model, images, labels, optimizer)
    # correct_num = correct_num_batch(labels, prediction)
    train_acc_metric.update_state(labels, prediction)

    # sum_correct_num += correct_num
    if i % 200 == 0 and i != 0:
      print('accuracy: {:.6f}, l2 loss: {:.6f}'.format(train_acc_metric.result(),
                                                       loss))
  train_acc = train_acc_metric.result()
  print("Training acc over epoch: %.4f" % (float(train_acc),))
  train_acc_metric.reset_states()

if __name__ == '__main__':
  strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())  
  model_info = get_model(model_list, MODEL_NAME)
  print(model_info)

  train_data, test_data = eval(model_info['data_set'])

  if ENABLE_MULTI_DEVICE:
    with strategy.scope():
      model = load_model()
  else:
    model = load_model()

  model.summary()

  # ==== Initial Dump ==== #
  # weights = [weight for layer in model.layers for weight in layer.weights]
  # dump_tensors(weights, "before")
  # =====Model fit() Training ===== #
  # model = compile_model(model)
  # train_steps = int(c.train_num / BATCH_SIZE)
  # val_steps = int(c.test_num / BATCH_SIZE)
  # model.fit(train_data, epochs=1, 
  #     steps_per_epoch=train_steps, validation_steps=val_steps, 
  #     validation_data=test_data, shuffle=False)
  # ===== Customize Training ===== #
  train(model, train_data)
  dump_tensors(weights, "after")

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
