import cv2
import numpy as np
import os
from imread_from_url import imread_from_url

from mobilestereonet import MobileStereoNet, CameraConfig
from mobilestereonet.utils import draw_disparity

from contextlib import contextmanager
import time
import logging


@contextmanager
def _log_time_usage(prefix=""):
	'''log the time usage in a code block
	prefix: the prefix text to show
	'''
	start = time.time()
	try:
		yield
	finally:
		end = time.time()
		elapsed_seconds = float("%.2f" % (end - start))
		logging.debug('%s: elapsed seconds: %s', prefix, elapsed_seconds)
		print ('%s: elapsed seconds: %s', prefix, elapsed_seconds)


class depth_estimation_inference:
	def __init__(self, model_path, camera_config=None):
		# Initialize model
		if camera_config is not None:
			self.mobile_depth_estimator = MobileStereoNet(
				model_path, camera_config=camera_config)
		else:
			self.mobile_depth_estimator = MobileStereoNet(model_path)

	def forward(self, left_image: cv2.Mat, right_image: cv2.Mat):
		# Estimate the depth
		disparity_map = self.mobile_depth_estimator(left_image, right_image)
		return disparity_map

def read_stereo_pairs_from_folder(folder_path, extension = '.png'):
	stereo_pair_paths = []
	for filename in os.listdir(folder_path):
		if extension in filename:
			if filename[-8:-4] == 'left':
				left_image_name = filename
				right_image_name = filename[:-8]+filename[-8:-4].replace('left', 'right')+extension
				if os.path.exists(folder_path+'/'+right_image_name):
					stereo_pair_paths.append((folder_path+'/'+left_image_name, folder_path+'/'+right_image_name))
	return stereo_pair_paths


if __name__ == '__main__':
	
	model_path = "models/model_528_240_float32.onnx"
	input_folder_path = "/home/aboggaram/data/Octiva/stereo_test_raw_images"
	depth_inference_object = depth_estimation_inference(model_path=model_path)
	
	# Load images
	stereo_pair_paths = read_stereo_pairs_from_folder(input_folder_path)
	left_img_path, right_img_path = stereo_pair_paths[0]
	left_img = cv2.imread(left_img_path)
	right_img = cv2.imread(right_img_path)

	with _log_time_usage("depth inference 1 pair: "):
		disparity_map = depth_inference_object.forward(left_image = left_img, right_image = right_img)
	
	color_disparity = draw_disparity(disparity_map)
	color_disparity = cv2.resize(color_disparity, (left_img.shape[1],left_img.shape[0]))

	combined_image = np.hstack((left_img, right_img, color_disparity))


	cv2.imwrite("out.jpg", combined_image)

	cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)	
	cv2.imshow("Estimated disparity", combined_image)
	cv2.waitKey(0)

	cv2.destroyAllWindows()
	del depth_estimation_inference

 