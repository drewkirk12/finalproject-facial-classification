import io
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import hyperparameters as hp
import sklearn

def plot_to_image(figure):
    """ Converts a pyplot figure to an image tensor. """

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image

class ConfusionMatrixLogger(tf.keras.callbacks.Callback):
    """ Keras callback for logging a confusion matrix for viewing
    in Tensorboard. """

    def __init__(self, logs_path, datasets):
        super(ConfusionMatrixLogger, self).__init__()

        self.datasets = datasets
        self.logs_path = logs_path

    def on_epoch_end(self, epoch, logs=None):
        self.log_confusion_matrix(epoch, logs)

    def log_confusion_matrix(self, epoch, logs):
        """ Writes a confusion matrix plot to disk. """

        test_pred = []
        test_true = []
        count = 0
        
        for img, label in self.datasets.test_data:
            pred = self.model(img, training=False)
            pred = tf.argmax(pred, axis=-1)
            test_pred.append(pred)
            test_true.append(label)
            print(pred.shape, end=' ', flush=True)
            count += 1
            if count >= 7100 / hp.batch_size:
                break

        print(tf.shape(test_pred[0]))
        print(tf.shape(test_true[0]))
        test_pred = tf.concat(test_pred, axis=0)
        test_true = tf.concat(test_true, axis=0)

        # Source: https://www.tensorflow.org/tensorboard/image_summaries
        cm = sklearn.metrics.confusion_matrix(test_true, test_pred, )

        for figsize in [3, 5, 8]:
            figure = self.plot_confusion_matrix(
                cm, class_names=self.datasets.classes,
                figsize=(figsize, figsize))
            cm_image = plot_to_image(figure)
            figure.savefig(f'confusion{figsize}.pdf')
            figure.savefig(f'confusion{figsize}.png')

        file_writer_cm = tf.summary.create_file_writer(
            self.logs_path + os.sep + "confusion_matrix")

        with file_writer_cm.as_default():
            tf.summary.image(
                "Confusion Matrix (on validation set)", cm_image, step=epoch)

    def plot_confusion_matrix(self, cm, class_names, figsize=(8, 8)):
        """ Plots a confusion matrix returned by
        sklearn.metrics.confusion_matrix(). """

        cm = np.around(cm.astype('float') / cm.sum(axis=1)
               [:, np.newaxis], decimals=2)

        # Source: https://www.tensorflow.org/tensorboard/image_summaries
        figure = plt.figure(figsize=figsize) # 5 5 
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)


        threshold = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center", color=color)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        return figure
    

class CustomModelSaver(tf.keras.callbacks.Callback):
    """ Custom Keras callback for saving weights of networks. """

    def __init__(self, checkpoint_dir, model, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.max_num_weights = max_num_weights

    def on_epoch_end(self, epoch, logs=None):
        """ At epoch end, weights are saved to checkpoint directory. """

        min_acc_file, max_acc_file, max_acc, num_weights = \
            self.scan_weight_files()

        cur_acc = logs["val_sparse_categorical_accuracy"]

        # Only save weights if test accuracy exceeds the previous best
        # weight file
        if cur_acc > max_acc:
            save_name = "weights.e{0:03d}-acc{1:.4f}.h5".format(
                epoch, cur_acc)
            if self.model.name == "se_res_net":
                save_location = self.checkpoint_dir + os.sep + "seresnet." + save_name
                print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                       "maximum TEST accuracy.\nSaving checkpoint at {location}")
                       .format(epoch + 1, cur_acc, location = save_location))
                self.model.save_weights(save_location)
            elif self.model.name == 'vgg_model':
                save_location = self.checkpoint_dir + os.sep + "vgg." + save_name
                print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                       "maximum TEST accuracy.\nSaving checkpoint at {location}")
                       .format(epoch + 1, cur_acc, location = save_location))
                # Only save weights of classification head of VGGModel
                self.model.save_weights(save_location)
            else:
                save_location = self.checkpoint_dir + os.sep + "inception." + save_name
                print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                       "maximum TEST accuracy.\nSaving checkpoint at {location}")
                       .format(epoch + 1, cur_acc, location = save_location))
                # Only save weights of classification head of InceptionModel
                self.model.save_weights(save_location)

            # Ensure max_num_weights is not exceeded by removing
            # minimum weight
            if self.max_num_weights > 0 and \
                    num_weights + 1 > self.max_num_weights:
                os.remove(self.checkpoint_dir + os.sep + min_acc_file)
        else:
            print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) DID NOT EXCEED "
                   "previous maximum TEST accuracy.\nNo checkpoint was "
                   "saved").format(epoch + 1, cur_acc))


    def scan_weight_files(self):
        """ Scans checkpoint directory to find current minimum and maximum
        accuracy weights files as well as the number of weights. """

        min_acc = float('inf')
        max_acc = 0
        min_acc_file = ""
        max_acc_file = ""
        num_weights = 0

        files = os.listdir(self.checkpoint_dir)

        for weight_file in files:
            if weight_file.endswith(".h5"):
                num_weights += 1
                file_acc = float(re.findall(
                    r"[+-]?\d+\.\d+", weight_file.split("acc")[-1])[0])
                if file_acc > max_acc:
                    max_acc = file_acc
                    max_acc_file = weight_file
                if file_acc < min_acc:
                    min_acc = file_acc
                    min_acc_file = weight_file

        return min_acc_file, max_acc_file, max_acc, num_weights
    
    
def get_activations(model, images, layer_names):
    outputs = []
    input_to_use = None
    
    if model.name == 'se_res_net':
        for name in layer_names:
            layer_output = model.get_layer(name).output
            outputs.append(layer_output)
        input_to_use = model.input
    elif model.name == 'vgg_model':
        for name in layer_names:
            layer_output = model.get_layer('vgg_base').get_layer(name).output
            outputs.append(layer_output)
        input_to_use = model.get_layer('vgg_base').input
    elif model.name == 'inception_model':
        for name in layer_names:
            layer_output = model.get_layer('inception_v3').get_layer(name).output
            outputs.append(layer_output)
        input_to_use = model.get_layer('inception_v3').input
        # Resize images for Inception model
        images = tf.image.resize(images, (299, 299))

    # Create a model that will return these outputs given the model input
    activation_model = tf.keras.models.Model(inputs=input_to_use, outputs=outputs)

    # Execute the model to get the activations
    return activation_model.predict(images)


def plot_activations(original_images, activations, layer_names):
    for i, layer_activations in enumerate(activations):
        fig, axes = plt.subplots(1, len(original_images), figsize=(20, 3))
        fig.suptitle(f"Layer: {layer_names[i]}", fontsize=16)

        for img_idx, img in enumerate(original_images):
            ax = axes[img_idx] if len(original_images) > 1 else axes
            img = np.squeeze(img)
            activation = layer_activations[img_idx, :, :, np.argmax(np.mean(layer_activations[img_idx], axis=(0, 1)))]
            activation = (activation - activation.min()) / (activation.max() - activation.min())
            activation = np.clip(activation, 0, 1)  # Ensuring all values are between 0 and 1

            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.imshow(activation, cmap='jet', alpha=0.5, interpolation='bilinear')
            ax.axis('off')

        plt.savefig(f'layer_{layer_names[i]}.png', bbox_inches='tight')
        plt.close(fig)
