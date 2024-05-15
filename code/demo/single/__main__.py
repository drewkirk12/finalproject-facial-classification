import argparse

import demo
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def parse_args():
	parser = argparse.ArgumentParser(
		description="Facial Emotion Recognition",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('input',
					 type=argparse.FileType('rb'),
					 help="Image to open")
	
	parser.add_argument('--output',
					 type=str,
					 help="Destination to write the plot",
					 required=False)
	
	return parser.parse_args()


def custom_preprocess(image, target_size):
	"""Custom preprocessing to load image files"""
	image = np.asarray(image)
	# Add color channels
	if image.ndim == 2:
		image = np.expand_dims(image, -1)

	# Convert to RGB
	if np.shape(image)[-1] == 1:
		image = np.broadcast_to(image, [*(np.shape(image)[:2]), 3])

	# Resize images to the required size
	image = demo.CropAndResize(target_size)(image)

	return image

def main():
	args = parse_args()

	# Fetch the models from the demo
	model_keys = ['seresnet2', 'inception', 'vgg']
	models = [demo.models[k] for k in model_keys]

	# Create the figure
	figure = plt.figure(figsize=(6, 4), layout='constrained')
	gs_leftright = figure.add_gridspec(1, 3)
	gs_preds = gs_leftright[1:3].subgridspec(len(models), 1)

	# Plot images on the left and predictions on the right
	image_axis = figure.add_subplot(gs_leftright[0])
	prediction_axes = [figure.add_subplot(gs_preds[i]) for i, _ in enumerate(models)]

	# Initialize axes
	image_artist = demo.create_image_plot(image_axis, 'Input image')
	prediction_plots = [demo.create_prediction_plot(ax,
												 model['labels'],
												 model['name'],
												 max=1.25) for model, ax in zip(models, prediction_axes)]

	with Image.open(args.input) as image:
		min_size = min(image.width, image.height)
		image_artist.set_data(custom_preprocess(image, (min_size, min_size)))

	for model, plot in zip(models, prediction_plots):
		# Instantiate the model
		predict = model['model']()
		input_size = model['input_size']
		preprocess = model['preprocess']
		class_labels = model['labels']

		# Load checkpoints
		predict(tf.keras.Input(shape=(*input_size, 3)))
		checkpoint_weights = model['checkpoint_weights']
		predict.load_weights(checkpoint_weights, by_name=False)

		# Make and plot a prediction
		artist, labels = plot
		with Image.open(args.input) as image:
			image = custom_preprocess(image, input_size)
			image = np.expand_dims(image, 0)
			image = preprocess(image)
			result = predict(image)[0]

			demo.plot_predictions(result, class_labels,
						 artist, labels)

	# Save the figure
	figure.show()
	if args.output is not None:
		figure.savefig(args.output)
	plt.show()




if __name__ == '__main__':
	main()