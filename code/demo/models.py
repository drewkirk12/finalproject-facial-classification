from models import InceptionModel, VGGModel
from models2 import SEResNet as SEResNet2
from preprocess import Datasets
import tensorflow as tf

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
		'name': 'VGG-16',
		'model': VGGModel,
		'input_size': (224, 224), # 244?
		'preprocess': tf.keras.applications.vgg16.preprocess_input,
		'checkpoint_weights': 'checkpoints/vgg_model/upload/vgg.weights.e011-acc0.6406.h5',
		'labels': class_labels_sorted,
		'visualize_layers': ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'],
	},
	'inception': {
		'name': 'Inception v3',
		'model': InceptionModel,
		'input_size': (299, 299),
		'preprocess': tf.keras.applications.inception_v3.preprocess_input,
		'checkpoint_weights': 'checkpoints/inception_model/upload/inception.weights.e010-acc0.6780.h5',
		'labels': class_labels_sorted,
		'visualize_layers': ['conv2d_1', 'activation_1', 'mixed0', 'mixed3', 'mixed7'],
	},
	'seresnet2': {
		'name': 'SE-ResNet-18',
		'model': lambda: SEResNet2(num_classes=len(class_labels)),
		'input_size': (224, 224),
		'preprocess': Datasets('../data', 'seresnet').preprocess_fn,
		'checkpoint_weights': 'checkpoints/seresnet2_model/upload/seresnet2.weights.e019-acc0.6305.h5',
		'labels': class_labels,
		'visualize_layers': ['conv1', 'seblock1', 'seblock2', 'seblock3', 'seblock4'],
	},
}
