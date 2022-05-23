import os
from pprint import pprint
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Read checkpoint variable')

# dataset and model
parser.add_argument('--ckpt', type=str, default='imagenet',
                    help='Checkpoint path')

args = parser.parse_args()


tf_path = os.path.abspath(args.ckpt)  # Path to our TensorFlow checkpoint
tf_vars = tf.train.list_variables(tf_path)
pprint(tf_vars)