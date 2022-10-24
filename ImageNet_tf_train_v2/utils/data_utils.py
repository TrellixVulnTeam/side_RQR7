import tensorflow as tf
import numpy as np
import os
import cv2
import config as c
import glob

"""
|------|
|      | height, y
|      |
|------|
 width, x
"""

def normalize(image):
  for i in range(3):
    image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]
  return image

def resize(image):
  return cv2.resize(image, (c.input_shape[0], c.input_shape[1]))

def load_list(list_path, image_root_path):
  images = []
  labels = []
  with open(list_path, 'r') as f:
    for line in f:
      line = line.replace('\n', '').split(' ')
      images.append(os.path.join(image_root_path, line[0]))
      labels.append(int(line[1]))
      
  return images, labels

def load_image(image_path, label):
  image = cv2.imread(image_path.numpy().decode()).astype(np.float32)

  image = resize(image)

  image = normalize(image)

  label_one_hot = np.zeros(c.category_num).astype(np.float32)
  label_one_hot[label] = 1.0

  return image, label_one_hot

def data_gen(images, labels):
  i = 0
  for path in images:
    image = cv2.imread(path).astype(np.float32)
    image = cv2.resize(image, (c.input_shape[0], c.input_shape[1]))
    for i in range(3):
      image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]
    label_one_hot = np.zeros(c.category_num)
    label_one_hot[labels[i]] = 1.0
    yield image, label_one_hot
    i = i + 1

def load_dataset(list_path, data_path):
  images, labels = load_list(list_path, data_path)
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  # gen = data_gen(images, labels)
  # dataset = tf.data.Dataset.from_generator(
  #     gen,
  #     output_signature=(
  #         tf.TensorSpec(shape=(), dtype=tf.float32),
  #         tf.TensorSpec(shape=(), dtype=tf.float32)))
  dataset = dataset.shuffle(len(images))
  dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y], Tout=[tf.float32, tf.float32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return dataset

def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'image/height': tf.io.FixedLenFeature([], tf.int64),
      'image/width': tf.io.FixedLenFeature([], tf.int64),
      'image/colorspace': tf.io.FixedLenFeature([], tf.string),
      'image/channels': tf.io.FixedLenFeature([], tf.int64),
      'image/class/label': tf.io.FixedLenFeature([], tf.int64),
      'image/class/synset': tf.io.FixedLenFeature([], tf.string),
      'image/format': tf.io.FixedLenFeature([], tf.string),
      'image/filename': tf.io.FixedLenFeature([], tf.string),
      'image/encoded': tf.io.FixedLenFeature([], tf.string)
  }

  content = tf.io.parse_single_example(element, data)

  height = content['image/height']
  width = content['image/width']
  channels = content['image/channels']
  label = content['image/class/label']
  image_buffer = content['image/encoded']
  label = tf.one_hot(label, c.category_num)

  #Decode a JPEG-encoded image to a uint8 tensor.
  feature = tf.io.decode_jpeg(image_buffer, channels=3,)
  feature = tf.image.resize(feature, (c.input_shape[0], c.input_shape[1]))
  #normalize
  mean = tf.constant([[c.mean]], dtype=tf.float32)
  std = tf.constant([[c.std]], dtype=tf.float32)
  feature_norm = tf.math.divide(tf.math.subtract(feature, mean), std)
  return (feature_norm, label)

def load_tfr_dataset(tfr_dir: str = "/content/", pattern: str = "*images.tfrecords"):
  files = glob.glob(os.path.join(tfr_dir, pattern), recursive=False)

  #create the dataset
  dataset = tf.data.TFRecordDataset(files)

  #pass every single feature through our mapping function
  dataset = dataset.map(parse_tfr_element)
  return dataset