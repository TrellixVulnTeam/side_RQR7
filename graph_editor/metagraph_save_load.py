# Build the model
...
with tf.Session() as sess:
  # Use the model
  ...
# Export the model to /tmp/my-model.meta.
meta_graph_def = tf.train.export_meta_graph(filename='/tmp/my-model.meta')

# Export the default running graph and only a subset of the collections.
meta_graph_def = tf.train.export_meta_graph(
    filename='/tmp/my-model.meta',
    collection_list=["input_tensor", "output_tensor"])

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
  new_saver.restore(sess, 'my-save-dir/my-model-10000')
  # tf.get_collection() returns a list. In this example we only want the
  # first one.
  train_op = tf.get_collection('train_op')[0]
  for step in xrange(1000000):
    sess.run(train_op)


varmap = {'model1/some_scope/weights': variable_in_model2,
          'model1/another_scope/weights': another_variable_in_model2}

saver = tf.train.Saver(varmap)
saver.restore(sess, path_to_saved_file)