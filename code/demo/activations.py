import tensorflow as tf

class ActivationVisualizer:
	def __init__(self, model, layer_name, preprocess):
		self.model = model
		self.layer_names = layer_name
		self.preprocess = preprocess

		input_to_use = None

		if model.name == 'se_res_net' or model.name == 'se_res_net_2':
			layer_output = model.get_layer(layer_name).output
			input_to_use = model.input
		elif model.name == 'vgg_model':
			layer_output = model.get_layer('vgg_base').get_layer(layer_name).output
			input_to_use = model.get_layer('vgg_base').input
		elif model.name == 'inception_model':
			layer_output = model.get_layer('inception_v3').get_layer(layer_name).output
			input_to_use = model.get_layer('inception_v3').input
		else:
			print(f'Unrecognized model {model.name}')
			layer_output = model.get_layer(layer_name).output
			input_to_use = model.input

		# Create a model that will return these outputs given the model input
		self.activation_model = tf.keras.models.Model(inputs=input_to_use, outputs=[layer_output])

	def __call__(self, image):
		# Feed image to the model
		image = tf.cast(image, tf.float32) # Necessary for inception
		images = tf.expand_dims(image, 0)
		images = self.preprocess(images)
		activation = self.activation_model(images)[0]
		activation = tf.norm(activation, axis=-1, keepdims=True)
		
		mean = tf.math.reduce_mean(activation, keepdims=True)
		std = tf.math.reduce_std(activation, keepdims=True)
		activation = 0.5 + (activation - mean) / std

		# Resize activation to the same size as the image
		activation = tf.image.resize(activation, image.shape[:2])
		activation = tf.broadcast_to(activation, (*activation.shape[:2], 3))

		activated_image = image / 255 * activation
		return tf.clip_by_value(activated_image, 0, 1)
