import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras_cv.layers import SqueezeAndExcite2D

# Hyperparameters (from paper):
num_epochs = 20
learning_rate = 0.01
momentum = 0.9
batch_size = 100
num_classes = 9

# TODO: Confirm SE Basic Block matches the paper
class SEBasicBlock(keras.layers.Layer):
    """ SE-ResNet model described in the paper. """
    def __init__(self, filters, kernel_size=3):
          super().__init__()
          self.filters = filters
          self.kernel_size=kernel_size

    def call(self, inputs):
        x = Conv2D(self.filters, self.kernel_size, 1, padding="same", activation="relu"),
        x = SqueezeAndExcite2D(self.filters)(x)
        return x

class SEResNet(tf.keras.Model):
    """ SE-ResNet model described in the paper. """

    def __init__(self):
        super(SEResNet, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.architecture = [
            Conv2D(64, 7, 1, padding="same", activation="relu", name='conv1'),
            SEBasicBlock(64, name='seblock1'),
            SEBasicBlock(128, name='seblock2'),
            SEBasicBlock(256, name='seblock3'),
            SEBasicBlock(512, name='seblock4'),
            Flatten(name='flatten'),
            Dense(num_classes, activation='softmax', name='dense')
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.categorical_crossentropy(labels, predictions)

# class SEResNet(tf.keras.Model):
#     """ SE-ResNet model described in the paper. """

#     def __init__(self):
#         super(SEResNet, self).__init__()

#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#         self.architecture = [
#             Conv2D(64, 7, 1, padding="same", activation="relu"),
#             SEBasicBlock(64),
#             SEBasicBlock(128),
#             SEBasicBlock(256),
#             SEBasicBlock(512),
#             Flatten(),
#             Dense(num_classes, activation='softmax')
#         ]

#     def call(self, x):
#         """ Passes input image through the network. """

#         for layer in self.architecture:
#             x = layer(x)
#         return x

#     @staticmethod
#     def loss_fn(labels, predictions):
#         """ Loss function for the model. """

#         return tf.keras.losses.categorical_crossentropy(labels, predictions)