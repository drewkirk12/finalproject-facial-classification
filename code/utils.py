import io
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import hyperparameters as hp

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
        for i in self.datasets.test_data:
            test_pred.append(self.model.predict(i[0]))
            test_true.append(i[1])
            count += 1
            if count >= 1500 / hp.batch_size:
                break

        test_pred = np.array(test_pred)
        test_pred = np.argmax(test_pred, axis=-1).flatten()
        test_true = np.array(test_true).flatten()

        # Source: https://www.tensorflow.org/tensorboard/image_summaries
        cm = sklearn.metrics.confusion_matrix(test_true, test_pred)
        figure = self.plot_confusion_matrix(
            cm, class_names=self.datasets.classes)
        cm_image = plot_to_image(figure)

        file_writer_cm = tf.summary.create_file_writer(
            self.logs_path + os.sep + "confusion_matrix")

        with file_writer_cm.as_default():
            tf.summary.image(
                "Confusion Matrix (on validation set)", cm_image, step=epoch)

    def plot_confusion_matrix(self, cm, class_names):
        """ Plots a confusion matrix returned by
        sklearn.metrics.confusion_matrix(). """

        # Source: https://www.tensorflow.org/tensorboard/image_summaries
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        cm = np.around(cm.astype('float') / cm.sum(axis=1)
                       [:, np.newaxis], decimals=2)

        threshold = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > threshold else "black"
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return figure
    

class CustomModelSaver(tf.keras.callbacks.Callback):
    """ Custom Keras callback for saving weights of networks. """

    def __init__(self, checkpoint_dir, task, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.task = task
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

            if self.task == '1':
                save_location = self.checkpoint_dir + os.sep + "your." + save_name
                print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                       "maximum TEST accuracy.\nSaving checkpoint at {location}")
                       .format(epoch + 1, cur_acc, location = save_location))
                self.model.save_weights(save_location)
            else:
                save_location = self.checkpoint_dir + os.sep + "vgg." + save_name
                print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                       "maximum TEST accuracy.\nSaving checkpoint at {location}")
                       .format(epoch + 1, cur_acc, location = save_location))
                # Only save weights of classification head of VGGModel
                self.model.head.save_weights(save_location)

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
    
    
def get_activations(model, input_data, layer_names):
    """ Fetches activations for a given model and input data for specified layers. """
    
    layer_outputs = [layer.output for layer in model.layers if layer.name in layer_names]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(input_data)
    return activations


def plot_activations(original_images, activations, layer_names):
    """ Overlays activation heatmaps on original images for specified layers. """
    
    for i, layer_activations in enumerate(activations):
        fig, axes = plt.subplots(1, len(original_images), figsize=(20, 3))
        fig.suptitle(f"Layer: {layer_names[i]}", fontsize=16)

        for img_idx, img in enumerate(original_images):
            ax = axes[img_idx]
            img = np.squeeze(img) # remove channel dimension if grayscale
            activation = layer_activations[img_idx, :, :, np.argmax(np.mean(layer_activations[img_idx], axis=(0, 1)))]
            
            ax.imshow(img, cmap='gray')
            ax.imshow(activation, cmap='jet', alpha=0.5, interpolation='bilinear')
            ax.axis('off')
        plt.show()
