# STATS 315: Lecture 19 (CNN IV)

## Convolutional Layers

### Local Connectivity
* For a neuron in the hidden layer:
    * Take input from patch
    * Compute weighted sum
    * Apply bias
1. Applying window of weights
2. Computing linear combinations
3. Activating with non-linear function

### Cross correlation + Padding + Stride
* Padding allows us to retain the shape of the original image/input, or even grow it with 0's
    * Generic method
    * What if with non-zero values?
* How do we move patch by patch?
    * Stride 1: Move by 1 pixel right until hitting boundary, then start again
    * Will patch overlap?
    