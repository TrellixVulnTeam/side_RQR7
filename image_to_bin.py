from os import walk
from PIL import Image
import sys
import numpy as np

def convertImageToBin(filename, w, h):
	image = Image.open(filename)
	image = image.resize((w, h), Image.ANTIALIAS)

	# data = np.array(image, dtype = np.uint8)
	
	data = np.array(image, dtype=np.float32)
	data = (data - 127.5) / 127.5

	data.tofile(filename.replace(".jpg", "") + "_" + str(w) + "_" + str(h) + ".bin")
	print ("Convert", filename, "success. Data:\n",data)


if __name__ == '__main__':
	print("input input_image, output_bin, w , h\n")
	convertImageToBin(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

