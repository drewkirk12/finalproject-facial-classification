# finalproject-facial-classification
Emotional Classification Models for Brown CS1430 Final Project

For data download refer to the file data/fer2013/README.md

In order to train the model run main.py with the --model argument set to any of the following ["seresnet", "seresnet2", "vgg", "Inception]

A previously trained set of model weights can be loaded using the --checkpoint argument and passing in the file path to the weights but this is not required if you'ld like to train from scratch

The --data argument allows for datasets to be specified using the filepath to the prefered dataset. This argument is not required in order to use the fer2013 dataset

Confusion matrices can be recorded by running a training epoch with the --confusion flag. This matrix will appear in tensorboard or a file path can be specified to save the matrix

In order to test the model the --evaluate flag

The --lime-explainer argument along with the path to an image can be input to obtain the LIME Images for that input

By using the --visualize-features map we can also obtain and store feature maps

## Data

We use FER2013. To load the data, download the dataset and extract it under data/fer2013. The files data/fer2013/train.csv and data/fer2013/fer2013/fer2013.csv should exist.

Dumitru, Ian Goodfellow, Will Cukierski, Yoshua Bengio. (2013). Challenges in Representation Learning: Facial Expression Recognition Challenge. Kaggle. https://kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge

## Demos

To run the live demo, enter the code/ directory, and run
```
python3 -m demo --model <model name> --camera <camera number>
```
`model name` must be either `vgg`, `inception`, or `seresnet2`.

Example:
```
python3 -m demo --model seresnet2 --camera 0
```

To show predictions for a single image, run
```
python3 -m demo.single <path to image>
```
