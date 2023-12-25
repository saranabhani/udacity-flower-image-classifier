# Udacity - AI Programming with Python: Create Your Own Image Classifier

This project is part of the Udacity AI Programming with Python Nanodegree. The project is centered around the development of an image classifier using PyTorch. The classifier is trained to recognize various species of flowers, which can potentially be integrated into a mobile app for real-time flower recognition. We use a dataset comprising 102 flower categories.

## Project Structure
- Part 1: Implementation of the classifier in a Jupyter notebook.
  - Load and preprocess the image dataset.
  - Train the image classifier.
  - Predict image content using the trained classifier.
- Part 2: Conversion of the model into a command-line application.

## Dataset
The dataset is divided into three parts: training, validation, and testing. The training process involves applying transformations like random scaling, cropping, and flipping to the images.

## Steps

### Data Preprocessing
- Load the image dataset.
- Apply transformations (scaling, cropping, flipping) for training.
  
### Development
- Load a pre-trained network (e.g., VGG).
- Define a new, untrained feed-forward network as a classifier with ReLU activations and dropout.
- Train the classifier layers using backpropagation.
- Track loss and accuracy on the validation set to optimize hyperparameters.
- Implement a predict function for classifying flower species in images.
- Convert the trained deep neural network into a usable command-line application.

## Usage
- Training:

  `python train.py data_dir --save_dir save_directory --arch "vgg13" --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu`

- Inference:

  `python predict.py /path/to/image checkpoint --top_k 3 --category_names cat_to_name.json --gpu`
