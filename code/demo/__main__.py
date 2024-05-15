import argparse
import cv2
import numpy as np
import tensorflow as tf

from demo.activations import ActivationVisualizer
from demo.camera import CameraImageProvider, CropAndResize
from demo.models import models
from demo.visualizer import Visualizer

def use_model(model, preprocess):
	"""Wrap a model to predict single images"""
	def predict(image):
		image = np.expand_dims(image, 0)
		image = preprocess(image)
		result = model(image)
		return result[0]
	return predict

def parse_args():
	parser = argparse.ArgumentParser(
		description="Facial Emotion Recognition",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"--model",
		type=str,
		default="seresnet2",
		help="Model to run. Options: seresnet2, vgg, inception.")
	parser.add_argument(
		"--camera",
		type=int, 
		default=0,
		help="Camera to open.")
	
	return parser.parse_args()


def main():
	args = parse_args()
	model_name = args.model
	camera_number = args.camera

	# Instantiate model
	model = models[model_name]['model']()
	input_size = models[model_name]['input_size']
	preprocess = models[model_name]['preprocess']
	class_labels = models[model_name]['labels']
	visualize_layers = models[model_name]['visualize_layers']
	model(tf.keras.Input(shape=(*input_size, 3)))
	checkpoint_weights = models[model_name]['checkpoint_weights']
	model.load_weights(checkpoint_weights, by_name=False)

	camera = cv2.VideoCapture(camera_number)
	if not camera.isOpened():
		raise RuntimeError(f'failed to open camera {camera_number}')

	camera_provider = CameraImageProvider(camera, filters=[
		lambda img: tf.image.central_crop(img, 1),
		CropAndResize(input_size)])
	modifiers = [(layer, ActivationVisualizer(model, layer, preprocess)) for layer in visualize_layers]

	visualizer = Visualizer(camera_provider, class_labels,
							classifier=use_model(model, preprocess),
							modifiers=modifiers)

	# Show with no delay
	visualizer.show(0)

	camera.release()

if __name__ == '__main__':
	main()