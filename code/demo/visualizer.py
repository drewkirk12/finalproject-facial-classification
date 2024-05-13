import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import skimage

zero_image = np.zeros((1, 1, 3))

class Visualizer:
	def __init__(self, image_provider, class_labels,
			  classifier,
			  default_color='navy',
			  highlight_color='red'):
		self.class_labels = class_labels
		self.get_image = image_provider
		self.predict = classifier
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
		self.modified_image_axis = figure.add_subplot(gridspec)
		self.modified_image_axis.set_xticks([])
		self.modified_image_axis.set_yticks([])
		self.modified_image_axis.set_title('Feature map')
		self.modified_image_artist = self.modified_image_axis.imshow(zero_image)

	def init_prediction_plot(self, figure, gridspec, class_labels):
		self.prediction_axis = figure.add_subplot(gridspec)
		self.prediction_artist = self.prediction_axis.barh(np.arange(len(class_labels)), np.ones((len(class_labels),)))
		self.prediction_labels = self.prediction_axis.bar_label(self.prediction_artist, class_labels)
		self.prediction_axis.set_xlim(0, 1.2)


	def update(self, frame):
		image = self.get_image()

		self.input_image_artist.set_data(image)

		# Modify image
		modified_image = image
		self.modified_image_artist.set_data(modified_image)

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


class CameraImageProvider:
	"""Provides images from a camera, processed by a given list of filters.
	"""
	def __init__(self, camera, filters=[]):
		self.camera = camera
		self.filters = filters

	def __call__(self):

		ok, image = self.camera.read()
		if not ok:
			raise RuntimeError('failed to capture frame')

		# Convert BGR to RGB
		image = image[...,::-1]

		# Convert to 0-255
		if np.max(image) <= 1:
			image = 255 * image

		# Run the image through other filters
		for filter in self.filters:
			image = filter(image)

		return image

class CropAndResize:
	"""Crop and resize an image to the given target size.
	"""
	def __init__(self, target_size):
		self.target_size = target_size

	def __call__(self, image):
		rows, cols, _ = np.shape(image)
		
		new_rows = np.round(cols * (self.target_size[0] / self.target_size[1]))
		new_cols = np.round(rows * (self.target_size[1] / self.target_size[0]))

		# Compute whether to crop rows or columns
		if new_rows < rows:
			new_cols = cols
		elif new_cols < cols:
			new_rows = rows

		# Calculate amount to remove from each dimension
		crop_rows = rows - new_rows
		crop_cols = cols - new_cols

		# Crop to new_rows, new_cols
		crop_amounts = [
			(int(crop_rows // 2), int((crop_rows + 1) // 2)),
			(int(crop_cols // 2), int((crop_cols + 1) // 2)),
			(0, 0),
		]
		image = skimage.util.crop(image, crop_amounts)
		# Rescale
		image = skimage.transform.resize(image, self.target_size)

		image = skimage.util.img_as_ubyte(image)

		return image
