import tensorflow as tf
import h5py
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
	help="output dir")
ap.add_argument("-i", "--input", required=True,
	help="input dir")
args = vars(ap.parse_args())

cpktLogFileName = args["input"] #cpkt 文件路径
with open(cpktLogFileName, 'r') as f:
	#权重节点往往会保留多个epoch的数据，此处获取最后的权重数据      
	cpktFileName = f.readline().split('"')[1]     
	h5FileName = './tmp.h5'
reader = tf.train.NewCheckpointReader(cpktFileName)
f = h5py.File(h5FileName, 'w')
t_g = None
for key in sorted(reader.get_variable_to_shape_map()):
	# 权重名称需根据自己网络名称自行修改
	if key.endswith('weights') or key.endswith('biases'):
		keySplits = key.split(r'/')
		keyDict = keySplits[1] + '/' + keySplits[1] + '/' + keySplits[2]
		print(keyDict)
		f[keyDict] = reader.get_tensor(key)
