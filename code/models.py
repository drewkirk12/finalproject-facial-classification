import tensorflow as tf
import hyperparameters as hp
import seresnet_hp as sern_hp
import keras.api._v2.keras as keras
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, multiply, Reshape
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

class SEBasicBlock(keras.layers.Layer):
    """ SE Basic Block as described in paper"""
    def __init__(self, filters, kernel_size=3, ratio=sern_hp.reduction_ratio, **kwargs):
              super(SEBasicBlock, self).__init__(**kwargs)
              self.C = filters
              self.kernel_size = kernel_size
              self.r = ratio
              # Stride length must be set to 2 to reduce params
              self.conv2d = Conv2D(self.C, self.kernel_size, 2, padding="same", activation="relu")
              self.pool2d = GlobalAveragePooling2D()
              self.fc1 = Dense(self.C//self.r, activation='relu')
              self.fc2 = Dense(self.C, activation='sigmoid')

    def call(self, inputs):
              in_block = self.conv2d(inputs)
              x = self.pool2d(in_block)
              x = self.fc1(x)
              x = self.fc2(x)
              return multiply([in_block, x])
    
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

    def __init__(self):
        super(SEResNet, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=sern_hp.learning_rate)

        self.architecture = [
            Conv2D(64, 7, 1, padding="same", activation="relu", name='conv1'),
            SEBasicBlock(64, name='seblock1'),
            SEBasicBlock(128, name='seblock2'),
            SEBasicBlock(256, name='seblock3'),
            SEBasicBlock(512, name='seblock4'),
            Flatten(name='flatten'),
            Dense(hp.num_classes, activation='softmax', name='dense')
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)
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
                     layer.trainable = False

              # I've seen multiple heads similar to this one for InceptionV3
              self.head = [
                     GlobalAveragePooling2D(),
                     Flatten(),
                     Dense(1024, activation='relu'),
                     Dropout(0.5),
                     Dense(hp.num_classes, activation='softmax')]

              self.inception = tf.keras.Sequential(self.inception, name="inception_base")
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
