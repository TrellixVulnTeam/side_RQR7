# import the necessary packages
from __future__ import print_function
# from imutils.video import VideoStream
import numpy as np
import argparse
# import imutils
import time
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-i", "--input", required=True,
	help="path of input video file")
ap.add_argument("-f", "--fps", type=int, default=None,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video, In OSX: MJPG (.mp4), DIVX (.avi), X264 (.mkv).")
args = vars(ap.parse_args())

def VideoTrans():
	# initialize the video stream and allow the camera
	# sensor to warmup
	print("[INFO] reading video...")
	print(args["codec"])
	cap = cv2.VideoCapture(args["input"])
	# cap = cv2.VideoCapture(args.input)
	if (cap.isOpened()== False): 
	    raise ValueError('!!! Error opening video stream or file !!!')
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	if args["fps"] is None:
		args["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
		pass

	# vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
	# time.sleep(2.0)
	# initialize the FourCC, video writer, dimensions of the frame, and
	# zeros array
	fourcc = cv2.VideoWriter_fourcc(*args["codec"])

	# store the image dimensions, initialize the video writer,
	# and construct the zeros array
	(h, w) = (height, width)
	writer = cv2.VideoWriter(args["output"], fourcc, args["fps"],
							 (w, h), True)
	# zeros = np.zeros((h, w), dtype="uint8")


	# loop over frames from the video stream
	while cap.isOpened():
		# grab the frame from the video stream and resize it to have a
		# maximum width of 300 pixels
		ret, frame = cap.read()
		# frame = imutils.resize(frame, width=300)

		if ret == True:
		# break the image into its RGB components, then construct the
		# RGB representation of each frame individually
		# (B, G, R) = cv2.split(frame)
		# R = cv2.merge([zeros, zeros, R])
		# G = cv2.merge([zeros, G, zeros])
		# B = cv2.merge([B, zeros, zeros])
		# construct the final output frame, storing the original frame
		# at the top-left, the red channel in the top-right, the green
		# channel in the bottom-right, and the blue channel in the
		# bottom-left
		# output = np.zeros((h * 2, w * 2, 3), dtype="uint8")
		# output[0:h, 0:w] = frame
		# output[0:h, w:w * 2] = R
		# output[h:h * 2, w:w * 2] = G
		# output[h:h * 2, 0:w] = B
		# write the output frame to file
			writer.write(frame)
		# show the frames
			cv2.imshow("Frame", frame)
			# cv2.imshow("Output", output)
			# if the `q` key was pressed, break from the loop
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
		else:
			break
	# do a bit of cleanup
	print("[INFO] cleaning up...")
	cv2.destroyAllWindows()
	cap.release()
	writer.release()

def main():

    VideoTrans()

    print('Translate finished !')


if __name__ == '__main__':
   main()