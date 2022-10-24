import tensorflow as tf

def cross_entropy_batch(y_true, y_pred):
  cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
  cross_entropy = tf.reduce_mean(cross_entropy)
  return cross_entropy

def accuracy(y_true, y_pred):
  correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
  accuracy = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
  return accuracy

def correct_num_batch(y_true, y_pred):
  correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
  correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
  return correct_num

def l2_loss(model, weight_decay=1e-4):
  variable_list = []
  for v in model.trainable_variables:
    if 'kernel' in v.name:
      variable_list.append(tf.nn.l2_loss(v))
  return tf.add_n(variable_list) * weight_decay
