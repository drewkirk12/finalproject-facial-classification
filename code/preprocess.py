import os
import random

import keras
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds

import fer2013
import hyperparameters as hp

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    # Data augmentation compatible with tf.Datasets
    augment_dataset = keras.Sequential([
        keras.layers.RandomRotation(30 / 360),
        ])

    def __init__(self, dataset, model, data_path=None, augment=augment_dataset):

        self.data_path = data_path
        self.model = model

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        if dataset == 'fashion_mnist':
            self.train_data, self.test_data = tfds.load(
                'fashion_mnist',
                split=['train', 'test'],
                as_supervised=True)
        elif dataset == 'fer2013':
            self.train_data = fer2013.load(os.path.join(self.data_path, "fer2013/train.csv"),
                as_supervised=True)
            self.test_data = fer2013.load(os.path.join(self.data_path, "fer2013/fer2013/fer2013.csv"), usage='PublicTest',
                as_supervised=True)
        else:
            raise ValueError(f'Unrecognized dataset {dataset}')

        # Mean and std for standardization
        self.mean = tf.zeros((hp.img_size,hp.img_size,3))
        self.std = tf.ones((hp.img_size,hp.img_size,3))
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
            (hp.preprocess_sample_size, hp.img_size, hp.img_size, 3))
        
        preprocess_data = self.train_data.take(hp.preprocess_sample_size)

        # Import images
        for i, example in enumerate(preprocess_data):
            img = example[0]
            img = tf.image.resize(img, (hp.img_size, hp.img_size))
            img = tf.cast(img, dtype=tf.float32) / 255.

            # Grayscale -> RGB
            if np.shape(img)[-1] == 1:
                img = tf.broadcast_to(img, (hp.img_size, hp.img_size, 3))

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

    def preprocess(self, data, shuffle=False, augment=augment_dataset):

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
            images = augment(images)
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
        if augment is not None:
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


