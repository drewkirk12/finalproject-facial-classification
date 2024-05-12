import tensorflow as tf
import hyperparameters as hp
import seresnet_hp as sern_hp
# import keras.api._v2.keras as keras
import keras as keras
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, multiply, ReLU, Reshape
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, Sequential

# conda install keras-cv -n cs1430
# from keras_cv.layers import SqueezeAndExcite2D

# Note that reduction ratio is not specified !! Will require experimentation

# class SEBasicBlock(keras.layers.Layer):
#     """ This version has too many params."""
#     def __init__(self, filters, kernel_size=3):
#           super().__init__()
#           self.filters = filters
#           self.kernel_size = kernel_size
#           self.conv2d = Conv2D(self.filters, self.kernel_size, 1, padding="same", activation="relu")
#           self.se = SqueezeAndExcite2D(self.filters)

#     def call(self, inputs):
#         x = self.conv2d(inputs)
#         x = self.se(x)
#         return x

class Downsample(keras.layers.Layer):
    def __init__(self, filters, strides, **kwargs):
        super().__init__(**kwargs)
        self.downsample = Conv2D(filters,
                kernel_size=1,
                strides=strides)

    def call(self, x):
        x = self.downsample(x)
        return x

class Residual(keras.layers.Layer):
    """ResNet basic block"""
    def __init__(self, filters,
            kernel_size=3,
            strides=1,
            **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2D(filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                activation='relu',
                use_bias=False)
        self.conv2 = Conv2D(filters,
                kernel_size=kernel_size,
                strides=1,
                padding='same',
                use_bias=False)
        self.downsample = Downsample(filters, strides)
        self.relu = ReLU()

    def call(self, x):
        identity = self.downsample(x)
        # Apply both convolution layers
        x = self.conv1(x)
        x = self.conv2(x)
        # Residual
        x += identity
        # Activation
        x = self.relu(x)
        return x

class SqueezeExcite(keras.layers.Layer):
    """Squeeze and Excite block"""
    def __init__(self, filters, ratio=sern_hp.reduction_ratio, **kwargs):
        super().__init__(**kwargs)

        self.pool2d = GlobalAveragePooling2D()
        self.fc1 = Dense(filters//ratio, activation='relu')
        self.fc2 = Dense(filters, activation='sigmoid')

    def call(self, x):
        identity = x
        x = self.pool2d(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = multiply([identity, x])
        return x


class SEBasicBlock(keras.layers.Layer):
    """ SE Basic Block as described in paper"""
    def __init__(self, filters, kernel_size=3, strides=1,
            ratio=sern_hp.reduction_ratio, **kwargs):
            super(SEBasicBlock, self).__init__(**kwargs)
            self.C = filters
            self.kernel_size = kernel_size
            self.r = ratio

            self.downsample = Downsample(filters, strides=strides)
            self.residual1 = Residual(filters, strides=strides)
            self.squeeze_excite1 = SqueezeExcite(filters)
            self.residual2 = Residual(filters, strides=1)
            self.squeeze_excite2 = SqueezeExcite(filters)

    def call(self, inputs):
        x = inputs
        # Apply first block
        identity1 = self.downsample(x)
        x = self.residual1(x)
        x = self.squeeze_excite1(x)
        x += identity1
        # Apply second block
        identity2 = x
        x = self.residual2(x)
        x = self.squeeze_excite2(x)
        x += identity2
        return x
    
##### Pytorch SEBlock implementation if needed #####
# import torch.nn as nn
# import torch

# class SE_Block(nn.Module):
#     def __init__(self, c, r=16):
#         super(SE_Block, self).__init__()
#         self.squeeze = nn.AdaptiveAvgPool2d(1)
#         self.excitation = nn.Sequential(
#             nn.Linear(c, c // r, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c // r, c, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         bs, c, _, _ = x.size()
#         y = self.squeeze(x).view(bs, c)
#         y = self.excitation(y).view(bs, c, 1, 1)
#         return x * y.expand_as(x)

class SEResNet(tf.keras.Model):
    """ SE-ResNet model described in the paper. """

    def __init__(self, num_classes, **kwargs):
        super(SEResNet, self).__init__(**kwargs)

        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=sern_hp.learning_rate, momentum=sern_hp.momentum)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

        self.architecture = tf.keras.Sequential([
            Conv2D(64, kernel_size=7, strides=2, padding='same',
                activation="relu", name='conv1',
                use_bias=False),
            MaxPool2D((3, 3), strides=2, padding='same', name='maxpool1'),
            SEBasicBlock(64, name='seblock1', strides=1),
            SEBasicBlock(128, name='seblock2', strides=2),
            SEBasicBlock(256, name='seblock3', strides=2),
            SEBasicBlock(512, name='seblock4', strides=2),
            GlobalAveragePooling2D(name='globalaveragepool'),
            Flatten(name='flatten'),
            Dense(num_classes, activation='softmax', name='dense1')
        ])

    def call(self, x):
        """ Passes input image through the network. """
        x = self.architecture(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

class VGGModel(tf.keras.Model):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        self.vgg16 = [
            # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1"),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]

        for layer in self.vgg16:
            layer.trainable = False

        self.head = [Flatten(),
                     Dropout(0.5),
                     Dense(hp.num_classes, activation='softmax')]

        # Don't change the below:
        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
        """ Passes the image through the network. """

        x = self.vgg16(x)
        x = self.head(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for model. """

        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    
class InceptionModel(tf.keras.Model):
       def __init__(self):
              super(InceptionModel, self).__init__()

              self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

              self.inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))

              for layer in self.inception.layers:
                     layer.trainable = True

              # I've seen multiple heads similar to this one for InceptionV3
              self.head = [
                     GlobalAveragePooling2D(),
                     Flatten(),
                     Dense(1024, activation='relu'),
                     Dropout(0.7),
                     Dense(hp.num_classes, activation='softmax')]

              # self.inception = tf.keras.Sequential(self.inception, name="inception_base")
              self.head = tf.keras.Sequential(self.head, name="inception_head")

       def call(self, x):
              """ Passes the image through the network. """

              x = self.inception(x)
              x = self.head(x)

              return x

       @staticmethod
       def loss_fn(labels, predictions):
              """ Loss function for model. """

              return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)


