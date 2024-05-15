from matplotlib import animation
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import numpy as np

zero_image = np.zeros((1, 1, 3))

def create_image_plot(axis, title=None):
	"""Create an empty plot to show images."""
	axis.set_xticks([])
	axis.set_yticks([])
	if title is not None:
		axis.set_title(title)
	return axis.imshow(zero_image)

def create_prediction_plot(axis, class_labels, title=None, max=1.2, label_padding=0):
	"""Create an empty plot to show predictions."""
	artist = axis.barh(np.arange(len(class_labels)), np.ones((len(class_labels),)))
	labels = axis.bar_label(artist, class_labels, padding=label_padding)
	if title is not None:
		axis.set_title(title)
	axis.set_xlim(0, max)
	return artist, labels

def plot_predictions(predictions, class_labels,
					 artist, labels,
					 default_color='navy',
					 highlight_color='red'):

	max_prediction_value = np.max(predictions)
	for i, label in enumerate(class_labels):
		artist.patches[i].set_width(predictions[i])
		if predictions[i] == max_prediction_value:
			artist.patches[i].set(color=highlight_color)
			labels[i].set(color=highlight_color)
		else:
			artist.patches[i].set(color=default_color)
			labels[i].set(color=default_color)


class Visualizer:
	def __init__(self, image_provider, class_labels,
			  classifier,
			  modifiers=[('identity', lambda x: x)],
			  default_color='navy',
			  highlight_color='red'):
		self.class_labels = class_labels
		self.get_image = image_provider
		self.predict = classifier
		self.modifiers = modifiers
		self.init_plot(class_labels)

		self.default_color = default_color
		self.highlight_color = highlight_color

	def init_plot(self, class_labels):
		# Create a figure
		self.figure = plt.figure()
		gridspec = self.figure.add_gridspec(4, 2)
		
		self.init_input_image(self.figure, gridspec[0:3, 0])
		self.init_modified_image(self.figure, gridspec[0:3, 1])
		self.init_prediction_plot(self.figure, gridspec[3, 0:2], class_labels)

	def init_input_image(self, figure, gridspec):
		self.input_image_axis = figure.add_subplot(gridspec)
		self.input_image_artist = create_image_plot(self.input_image_axis, 'Input image')

	def init_modified_image(self, figure, gridspec):
		self.modified_image_axes = [None for _ in self.modifiers]
		self.modified_image_artists = [None for _ in self.modifiers]
		
		grid_size = int(np.ceil(len(self.modifiers) ** 0.5))
		num_rows = int(np.ceil(len(self.modifiers) / grid_size))
		subgridspec = GridSpecFromSubplotSpec(num_rows, grid_size, gridspec)
		
		for i, label_modifier in enumerate(self.modifiers):
			label, _ = label_modifier
			row = i // grid_size
			col = i % grid_size

			self.modified_image_axes[i] = figure.add_subplot(subgridspec[row, col])
			self.modified_image_artists[i] = create_image_plot(self.modified_image_axes[i], label)

	def init_prediction_plot(self, figure, gridspec, class_labels):
		self.prediction_axis = figure.add_subplot(gridspec)
		self.prediction_artist, self.prediction_labels = create_prediction_plot(self.prediction_axis,
																		  class_labels)


	def update(self, frame):
		image = self.get_image()

		self.input_image_artist.set_data(image)

		# Modify image (feature maps)
		for i, label_modify in enumerate(self.modifiers):
			_, modify = label_modify
			modified_image = modify(image)
			self.modified_image_artists[i].set_data(modified_image)

		prediction_values = self.predict(image)

		plot_predictions(prediction_values, self.class_labels,
					 self.prediction_artist, self.prediction_labels,
					 default_color=self.default_color,
					 highlight_color=self.highlight_color)


	def show(self, interval, save_count=0):
		ani = animation.FuncAnimation(fig=self.figure, func=self.update, interval=interval,
								save_count=save_count)
		plt.show()

