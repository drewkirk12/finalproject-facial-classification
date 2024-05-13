from matplotlib import animation
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import numpy as np

zero_image = np.zeros((1, 1, 3))

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
		self.input_image_axis.set_xticks([])
		self.input_image_axis.set_yticks([])
		self.input_image_axis.set_title('Input image')
		self.input_image_artist = self.input_image_axis.imshow(zero_image)

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
			self.modified_image_axes[i].set_xticks([])
			self.modified_image_axes[i].set_yticks([])
			self.modified_image_axes[i].set_title(label)
			self.modified_image_artists[i] = self.modified_image_axes[i].imshow(zero_image)

	def init_prediction_plot(self, figure, gridspec, class_labels):
		self.prediction_axis = figure.add_subplot(gridspec)
		self.prediction_artist = self.prediction_axis.barh(np.arange(len(class_labels)), np.ones((len(class_labels),)))
		self.prediction_labels = self.prediction_axis.bar_label(self.prediction_artist, class_labels)
		self.prediction_axis.set_xlim(0, 1.2)


	def update(self, frame):
		image = self.get_image()

		self.input_image_artist.set_data(image)

		# Modify image (feature maps)
		for i, label_modify in enumerate(self.modifiers):
			_, modify = label_modify
			modified_image = modify(image)
			self.modified_image_artists[i].set_data(modified_image)

		prediction_values = self.predict(image)

		max_prediction_value = np.max(prediction_values)
		for i, label in enumerate(self.class_labels):
			self.prediction_artist.patches[i].set_width(prediction_values[i])
			if prediction_values[i] == max_prediction_value:
				self.prediction_artist.patches[i].set(color=self.highlight_color)
				self.prediction_labels[i].set(color=self.highlight_color)
			else:
				self.prediction_artist.patches[i].set(color=self.default_color)
				self.prediction_labels[i].set(color=self.default_color)


	def show(self, interval, save_count=0):
		ani = animation.FuncAnimation(fig=self.figure, func=self.update, interval=interval,
								save_count=save_count)
		plt.show()

