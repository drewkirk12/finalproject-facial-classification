import numpy as np
import skimage

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
