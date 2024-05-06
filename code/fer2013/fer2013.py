import numpy as np

from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow  as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils.lazy_imports_utils import pandas as pd


emotions = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral',
    ]
image_size = (48, 48, 1)


def _to_image(pixels):
    arr = np.fromstring(pixels, dtype=np.uint8, sep=' ')
    arr = np.reshape(arr, image_size)
    return arr


def load(data_path, usage=None, as_supervised=False):
    """Load the given csv file with FER2013 data. If usage is specified, it
    should be one of "Training", "PublicTest", or "PrivateTest".
    """
    # Read data
    df = pd.read_csv(data_path)

    # Convert text pixels to images
    df['pixels'] = df['pixels'].apply(_to_image)
    has_label = 'emotion' in df.columns

    # Convert emotions to integers
    if has_label:
        df['emotion'] = df['emotion'].apply(int)

    # Select a specific split (Training, PublicTest, PrivateTest)
    if usage is not None:
        df = df[df['Usage'] == usage]
        
    # Convert relevant pixels to a NumPy array.
    pixels = df['pixels'].to_numpy()
    pixels = np.stack(pixels, axis=0)

    # Create a dataset from the images and labels
    if has_label:
        dataset_tensors = {
            'image': pixels,
            'label': df['emotion'],
        }
        if as_supervised:
            dataset_tensors = (dataset_tensors['image'],
                    dataset_tensors['label'])
    else:
        dataset_tensors = {
            'image': pixels,
        }
        if as_supervised:
            dataset_tensors = (dataset_tensors['image'],)

    dataset = tf.data.Dataset.from_tensor_slices(dataset_tensors)

    return dataset


if __name__ == '__main__':
    mnist_builder = tfds.builder("mnist")
    mnist_info = mnist_builder.info
    mnist_builder.download_and_prepare()
    datasets = mnist_builder.as_dataset()


    train_data = load('./data/fer2013/train.csv')
    test_data = load('./data/fer2013/fer2013/fer2013.csv', usage='PrivateTest')
    print(test_data.element_spec)
