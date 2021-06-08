# Tensorflow Graph

### pb import as Graph, 藉由sess轉成 GraphDef
#### GraphDef 才有 node 的 attr

```python3
pb_graph = tf.Graph()
with pb_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.input_pb, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
with tf.Session(graph=pb_graph, config=config) as sess:
    sess.graph_def.node
```

### pb import as GraphDef

```python3
pb_graph = tf.GraphDef()
with tf.gfile.GFile(args.input_pb, 'rb') as fid:
    serialized_graph = fid.read()
    pb_graph.ParseFromString(serialized_graph)
    print('='*10, 'Check out the input placeholders:', '='*10)
    nodes = [n.name + ' => ' +  n.op for n in pb_graph.node if n.op in ('Placeholder')]
    for node in nodes:
        print(node)
    tf.import_graph_def(pb_graph, name='')
```