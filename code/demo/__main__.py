import argparse
import cv2
import numpy as np
import os
import tensorflow as tf

from demo.camera import CameraImageProvider, CropAndResize
from demo.visualizer import Visualizer
from demo.activations import ActivationVisualizer
from models import InceptionModel, VGGModel
from models2 import SEResNet as SEResNet2
from preprocess2 import Datasets as Datasets2

class_labels = [
	'Angry',
	'Disgust',
	'Fear',
	'Happy',
	'Sad',
	'Surprise',
	'Neutral',
]
# Note: because of variations in training and testing, each model might have a
# different set of class labels.
class_labels_sorted = sorted(class_labels)

models = {
	'vgg': {
		'model': VGGModel,
		'input_size': (224, 224), # 244?
		'preprocess': tf.keras.applications.vgg16.preprocess_input,
		'checkpoint_weights': 'checkpoints/vgg_model/upload/vgg.weights.e011-acc0.6406.h5',
		'labels': class_labels_sorted,
		'visualize_layers': ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'],
	},
	'inception': {
		'model': InceptionModel,
		'input_size': (299, 299),
		'preprocess': tf.keras.applications.inception_v3.preprocess_input,
		'checkpoint_weights': 'checkpoints/inception_model/upload/inception.weights.e010-acc0.6780.h5',
		'labels': class_labels_sorted,
		'visualize_layers': ['conv2d_1', 'activation_1', 'mixed0', 'mixed3', 'mixed7'],
	},
	'seresnet2': {
		'model': lambda: SEResNet2(num_classes=len(class_labels)),
		'input_size': (224, 224),
		'preprocess': Datasets2('fer2013', 'seresnet', data_path='../data', augment=False).preprocess_fn,
		'checkpoint_weights': 'checkpoints/seresnet2_model/upload/seresnet2.weights.e019-acc0.6305.h5',
		'labels': class_labels,
		'visualize_layers': ['conv1', 'seblock1', 'seblock2', 'seblock3', 'seblock4'],
	},
}


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
		lambda img: tf.image.central_crop(img, 0.6),
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