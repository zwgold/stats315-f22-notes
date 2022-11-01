# STATS 315: Lecture 18 (CNN III)

## JITTs
* More layers, more neurons/higher layer sizes, and more data can reduce the BIAS of a neural network
* Counteract overfitting with regularization:
    * Weights will change LESS, due to nature of regularization (punishing weights)
    * Expect predictions to have a higher bias (lower variance)
* Regularization:
    * Restrict model complexity
        * Reduce some bias
    * Less dependent on training data
    * One of the ways to tradeoff bias and variance
* Learning Rate
    * Larger learning rate --> you'll potentially jump around the "valley"/local optima, and bounce around it
    * Smaller is better for optimization, but it takes slower, so more steps of computation

## Batch Normalization
* Batch Normalization
    * Helps gradient descent converge faster for NNs
    * NOT for regularization
    * Input normalization: Make all features 0 mean and std dev 1
        * Take the mean of all features and std dev of all features
        * $X_i = \frac{X_i - Mean_i}{StdDev_i}$
        * NOT LAYERED NORMALIZATION (Normalize across each sample for features)
    * Make all features on the same scale
    * Features on different scales can take longer to reach their optima/minima for optimization
    * Normalized data will help improve convergence speed
    * Same logic requiring us to normalize hte input for the first layer will aapply to each of the hidden layers
    * Inputs of each hidden layer are the activation from the previous later, and must also be normalized
    * Batch Normalization is just another layer that gets inserted between hidden layers
        * Job is to take output from the first hidden layer, normalize them, and then pass them as input
* Batch Layer as Parameters:
    * 2 learnable ones are beta and gamma
    * Two non-learnable
        * Mean Moving Average
        * Variance Moving Average
        * Both saved as "state" in the Batch Norm Layer
    * Each Batch Norm layer has its own copy of parameters
* Training
    * We feed the network one mini-batch of data at a time
    * During the forward pass, each layer of the network processes that mini-batch of data
* Testing
    * During it, we make predictions on a single sample, no mini batch
        * Inference
    * Two Moving Average parameters come in, the ones we calculated during training and are saved with the model
    * We use those saved mean and variance values for the batch norm during inference
* Can be done before or after activation
    * Putting it after usually works better

## CNNs

### CNNs for Classification
* Convolution: Apply filters with learned weights to generate feature maps
* Non-linearity: Often ReLU
* Pooling: Downsampling operation on each feature map
* Train model with image data, learn weights of filters in convolutional layers
* REFER to notes from last lecture