# LipNet

A replication of the LipNet architecture for deep-learning-based lipreading as presented in the [original paper](https://arxiv.org/abs/1611.01599).

This repository is part of a project for the course L545 - Computation and Linguistic Analysis at Indiana University Bloomington.

## Requirements

TBD

## Implementation

### Dataset
Like in the original paper, the model was trained on the [GRID audiovisual sentence corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/).
Due to computational limitations, this implementation was only trained on one speaker, S3.
As in the paper, 255 random videos were sampled to form the test set, with the rest used for training.

### Model

Because adopting the hyperparameters from the original paper resulted in severe underfitting when applied to only one speaker, the following changes were made:
- Kernel size was increased to (3, 3, 3).
- Padding was set to `same', so that the output of each convolutional layer had the same shape as the input.
- The input size for the first convolutional layer was changed to 75 x 1 x 40 x 60 to accommodate the reduction in color channels and smaller mouth region crop.
- The output channels for the convolutional layers were changed from from 32, 64, and 96 to 128, 256, and 75 respectively.
- The dropout layers were removed after the pooling layers, since overfitting was not a concern.
- The number of hidden nodes within the GRUs were halved from 256 to 128. This was done to prevent temporal overfitting, i.e., a case where the model only learns the sentence template, rather than what is actually being said (In a previous experiment, the model predicted the same string of characters for each video).

These changes are largely based on [Nicholas Renotte's implementation](https://github.com/nicknochnack/LipNet), which likewise only used a single speaker.

Because the paper does not state how many epochs the model was trained for, this implementation was run for 100 epochs.

### Evaluation Metrics

The model was evaluated based on Word Error Rate (WER) and Character Error Rate (CER), which are calculated by taking the edit distance between the ground truth and the predicted string on the word or character level respectively and dividing it by the number of words or characters in the ground truth.

## Results

On the test set, the model achieved a WER of 23.56% and a CER of 14.78%.
This is without using a language model in the CTC Decoder, which would further increase accuracy.


