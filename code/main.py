"""
CS1430 - Computer Vision
Brown University
Final Project: Facial Emotion Recognition
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
import numpy as np

import hyperparameters as hp
from models import SEResNet
from preprocess import Datasets

def parse_args():
    """ Parse command line arguments. """

    parser = argparse.ArgumentParser(
        description="Facial Emotion Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        default="seresnet",
        help="Model to use for training.")
    parser.add_argument(
        "--data_path",
        type=str, 
        default="data/",
        help="Path to data directory.")
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        "--evaluate",
        action='store_true',
        help="Evaluate model on test set.")
    


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0)
            # ,
        # ImageLabelingLogger(logs_path, datasets),
        # CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    # if ARGS.confusion:
    #     callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )    


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of main.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)

    # Run script from location of main.py
    os.chdir(sys.path[0])

    datasets = Datasets(ARGS.data, ARGS.task)

    if ARGS.model == 'seresnet':
        model = SEResNet()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "your_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "your_model" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
    """
    Add additional models here
    """
    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        if ARGS.model == 'seresnet':
            model.load_weights(ARGS.load_checkpoint, by_name=False)
        else:
            model.head.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if ARGS.evaluate:
        test(model, datasets.test_data)

        #Lime explanation
        #path = ARGS.lime_image
        #LIME_explainer(model, path, datasets.preprocess_fn, timestamp)
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)


# Make arguments global
ARGS = parse_args()

main()