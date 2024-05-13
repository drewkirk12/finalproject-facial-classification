import os
import random

import keras
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

import fer2013
import hyperparameters as hp

# Class for augmenting images
class ImageAugmenter:
    def __init__(self, image_size):
        self.random_rotation = tf.keras.layers.RandomRotation(30 / 360)
        self.random_crop = tf.keras.layers.RandomCrop(*image_size)
        self.image_size = image_size
        self.resized_image_size = \
                tf.cast(
                        tf.math.round(9/8 * tf.constant(image_size, dtype=tf.float32)),
                        dtype=tf.int32)

    @tf.function
    def __call__(self, image):
        image = tf.image.random_brightness(image, 0.2)
        image = self.random_rotation(image)
        image = tf.image.resize(image, self.resized_image_size)
        image = self.random_crop(image)
        return image

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, dataset, model, data_path=None, augment=True,
            target_shape=(224, 224, 3)):

        self.data_path = data_path
        self.model = model
        self.target_shape = target_shape

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        if dataset == 'fer2013':
            self.train_data = fer2013.load(os.path.join(self.data_path, "fer2013/train.csv"),
                as_supervised=True)
            self.test_data = fer2013.load(os.path.join(self.data_path, "fer2013/fer2013/fer2013.csv"), usage='PublicTest',
                as_supervised=True)
        else:
            self.train_data, self.test_data = tfds.load(
                dataset,
                split=['train', 'test'],
                as_supervised=True)

        self.augmenter = ImageAugmenter(target_shape[:2])

        # Mean and std for standardization
        self.mean = tf.zeros(self.target_shape)
        self.std = tf.ones(self.target_shape)
        self.calc_mean_and_std()

        self.train_data = self.preprocess(self.train_data, shuffle=True,
                augment=augment)
        self.test_data = self.preprocess(self.test_data, shuffle=True,
                augment=augment)


    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, *self.target_shape))
        
        preprocess_data = self.train_data.take(hp.preprocess_sample_size)

        # Import images
        for i, example in enumerate(preprocess_data):
            img = example[0]
            img = tf.image.resize(img, self.target_shape[:2])
            img = tf.cast(img, dtype=tf.float32) / 255.

            # Grayscale -> RGB
            img = tf.broadcast_to(img, self.target_shape)

            data_sample[i] = img.numpy()

        self.mean = tf.math.reduce_mean(data_sample, axis=0, keepdims=True)
        self.std = tf.math.reduce_std(data_sample, axis=0, keepdims=True)

        self.mean = tf.cast(self.mean, tf.float32)
        self.std = tf.cast(self.std, tf.float32)

        # Prevent NaN values when dividing by self.std
        self.std = self.std.numpy()
        self.std[self.std == 0] = 1
        self.std = tf.convert_to_tensor(self.std)

    @tf.function
    def standardize(self, img):
        """ Function for applying standardization to an input image. """
        if len(img.get_shape()) == 4:
            img = (img - self.mean) / self.std
        elif len(img.get_shape()) == 3:
            img = (img - self.mean[0]) / self.std[0]
        return img

    @tf.function
    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """
        # Grayscale -> RGB
        if img.get_shape()[-1] == 1:
            img = tf.concat([img, img, img], axis=-1)
        # Resize images
        img = tf.image.resize(img, (hp.img_size, hp.img_size))

        if self.model == 'vgg':
            img = tf.keras.applications.vgg16.preprocess_input(img)
        elif self.model == 'inception':
            img = tf.keras.applications.inception_v3.preprocess_input(img)
        else:
            img = tf.cast(img, tf.float32) / 255.
            # Standardize
            img = self.standardize(img)

        return img

    # Felicity's data augmentation (worked very well for HW5)
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
              rotation_range=5,
              width_shift_range=0.1,
              height_shift_range=0.1,
              shear_range=0.2,
              zoom_range=0.2,
              horizontal_flip=True,
              fill_mode='nearest',
              preprocessing_function=preprocess_fn)

    # Data augmentation as described in paper
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
              rotation_range=30,
              preprocessing_function=preprocess_fn)

    # Data augmentation compatible with tf.Datasets

    def preprocess(self, data, shuffle=False, augment=True):

        @tf.function
        def preprocess_images(images, labels):
            images = self.preprocess_fn(images)
            return images, labels

        # Felicity's data augmentation (worked very well for HW5)
        # data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        #         rotation_range=5,
        #         width_shift_range=0.1,
        #         height_shift_range=0.1,
        #         shear_range=0.2,
        #         zoom_range=0.2,
        #         horizontal_flip=True,
        #         fill_mode='nearest',
        #         preprocessing_function=self.preprocess_fn)

        @tf.function
        def augment_image_batch(images, labels):
            """Augment a batch of images and labels"""
            images = self.augmenter(images)
            return images, labels

        # Cache data. Must happen before augmenntation.
        # data = data.cache()

        # Shuffle data
        if shuffle:
            data = data.shuffle(64)

        # Create preprocessing batches
        data = data.batch(hp.batch_size)

        # Preprocess images
        data = data.map(preprocess_images, num_parallel_calls=tf.data.AUTOTUNE)

        # Data augmentation
        if augment:
            data = data.map(augment_image_batch, num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch data
        data = data.prefetch(tf.data.AUTOTUNE)

        return data


if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)
    datasets = Datasets('fer2013', 'seresnet', './data') 
    tfds.benchmark(datasets.train_data, batch_size=32)
    print('After caching')
    tfds.benchmark(datasets.train_data, batch_size=32)


