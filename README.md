# LipNet

- [Original Paper](https://arxiv.org/abs/1611.01599)

## Requirements

## Implementation

Because adopting the hyperparameters from the original paper resulted in severe underfitting when applied to only one speaker, the following changes were made:
- Kernel size was increased to (3, 3, 3).
- Padding was set to `same', so that the output of each convolutional layer had the same shape as the input.
- The input size for the first convolutional layer was changed to 75 x 1 x 40 x 60 to accommodate the reduction in color channels and smaller mouth region crop.
- The output channels for the convolutional layers were changed from from 32, 64, and 96 to 128, 256, and 75 respectively.
- The dropout layers were removed after the pooling layers, since overfitting was not a concern.
- The number of hidden nodes within the GRUs were halved from 256 to 128. This was done to prevent temporal overfitting, i.e., a case where the model only learns the sentence template, rather than what is actually being said.

These changes are largely based on [Nicholas Renotte's implementation](https://github.com/nicknochnack/LipNet), which likewise only used a single speaker.