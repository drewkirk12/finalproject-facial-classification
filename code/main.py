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

import hyperparameters as hp
from models import SEResNet
from models import VGGModel
from models import InceptionModel
from preprocess import Datasets
from utils import \
    ConfusionMatrixLogger, CustomModelSaver, get_activations, plot_activations

from skimage.io import imread
from skimage.transform import resize
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

def parse_args():
    """ Parse command line arguments. """

    parser = argparse.ArgumentParser(
        description="Facial Emotion Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        default="seresnet",
        help="Model to use for training. Options: seresnet, vgg, inception.")
    parser.add_argument(
        "--data",
        type=str, 
        default="../data/",
        help="Path to data directory.")
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        "--evaluate",
        action='store_true',
        help="Evaluate model on test set.")
    parser.add_argument(
        "--feature-maps",
        action='store_true',
        help="Visualize feature maps of the model.")
    parser.add_argument(
        '--lime-image',
        type=str,
        default=None,
        help='''Name of an image in the dataset to use for LIME evaluation.''')
    return parser.parse_args() 


def LIME_explainer(model, path, preprocess_fn, timestamp):
    """
    This function takes in a trained model and a path to an image and outputs 4
    visual explanations using the LIME model
    """

    save_directory = "lime_explainer_images" + os.sep + timestamp
    if not os.path.exists("lime_explainer_images"):
        os.mkdir("lime_explainer_images")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    image_index = 0

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):
        nonlocal image_index

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)

        image_save_path = save_directory + os.sep + str(image_index) + ".png"
        plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        plt.show()

        image_index += 1

    # Read the image and preprocess it as before
    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
    image = preprocess_fn(image)
    

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=1000)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True)

    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False)

    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False)

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")

    image_save_path = save_directory + os.sep + str(image_index) + ".png"
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.show()   

def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        #ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.model, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None, # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
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

    if ARGS.model == 'seresnet':
        model = SEResNet(7)
        input_shape = (224, 224, 3)
        checkpoint_path = "checkpoints" + os.sep + \
            "seresnet_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "seresnet_model" + \
            os.sep + timestamp + os.sep

        model(tf.keras.Input(shape=input_shape))
        # Print summary of model
        model.summary(expand_nested=True)

    """
    Add additional models here
    """
    
    if ARGS.model == "vgg":
            model = VGGModel()
            input_shape = (224, 224, 3)
            checkpoint_path = "checkpoints" + os.sep + \
                "vgg_model" + os.sep + timestamp + os.sep
            logs_path = "logs" + os.sep + "vgg_model" + \
                os.sep + timestamp + os.sep

            # Build model
            model(tf.keras.Input(shape=input_shape))
            # Print summary of model
            model.summary(expand_nested=True)
            # Load base of VGG model
            model.vgg16.load_weights('vgg16_imagenet.h5', by_name=True)

    if ARGS.model == "inception":
            model = InceptionModel()
            input_shape = (299, 299, 3)
            checkpoint_path = "checkpoints" + os.sep + \
                "inception_model" + os.sep + timestamp + os.sep
            logs_path = "logs" + os.sep + "inception_model" + \
                os.sep + timestamp + os.sep
            # Build model
            model(tf.keras.Input(shape=input_shape))
            # Print summary of model
            model.summary(expand_nested=True)

            # Load base of Inception model - leave commented out!!!
            # model.inception.load_weights('inception_v3_weight.h5', by_name=True)
    
    if model == None:
        raise RuntimeError(f'unrecognized model {ARGS.model}')


    # Load data
    dataset_name = 'fer2013'
    print(f'Loading data {dataset_name}')
    datasets = Datasets(dataset_name,
            ARGS.model, ARGS.data, augment=True,
            target_shape=input_shape)

    
    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)

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
        path = ARGS.lime_image
        if path is not None:
            LIME_explainer(model, path, datasets.preprocess_fn, timestamp)
    else:
        print("Training", ARGS.model,"model")
        train(model, datasets, checkpoint_path, logs_path, init_epoch)
        
    if ARGS.feature_maps:
        test_images, _ = next(iter(datasets.test_data))
        if ARGS.model == 'seresnet':
            layer_names_to_visualize = ['conv1', 'seblock1', 'seblock2', 'seblock3', 'seblock4']
        elif ARGS.model == 'vgg':
            layer_names_to_visualize = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
        elif ARGS.model == 'inception':
            layer_names_to_visualize = ['conv2d_1', 'activation_1', 'mixed0', 'mixed3', 'mixed7']
        activations = get_activations(model, test_images, layer_names_to_visualize)
        plot_activations(test_images, activations, layer_names_to_visualize)


# Make arguments global
ARGS = parse_args()

main()
