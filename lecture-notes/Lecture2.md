# STATS 315: Lecture 2

## Topics
* What is a neural network
* Intuition on structure
weights, biases, activations
* How it can be used

### Image Recognition
* For us, 3 is just 3
* When inputting to a computer as a number, it's also straightforward
* What about an image of the number 3? What about different ways to write it? Different resolutions and pixel counts?
* For Image Recognition: Takes in a GIANT matrix of all the RGB values, recognize the numbers based on matrix values
* Solution: Neural Networks can do it!

### Plain Vanilla NN (multilayer perceptron)

### Main Ingredients of NNs
* Neurons: "Thing that holds a number" (for now, this is wildly incorrect and this definition should be burned)
    * Data Value, the "Activation" ($[0,1]$)
    * Each Neuron holds value of pixels
* For our example of image classification: Note that the last layer outputs probabilities of the image being of each digit.
    * 0 holds probability that the image belongs to digit 0, and so on up to and including digit 9
* The layers go one direction, not in reverse!

### Hidden Layers
* How does 1 layer influence another? Why the layers?
* Hidden layers are the layers between the input and output layers.
* Takes in a set of weighted inputs and product output via activation function

### Dissection Image Recognition (Structure of the NN)
* We can partition the image into "submodules"/different portions such that
* The hidden layers will pick up the patterns, and note which ones correspond to different numbers,
* Recognizing upper loops could be just as hard, what could they correspond to?
    * Might need to dissect it even more
* Raw Pixels $\rightarrow$ Little Edges $\rightarrow$ Little Shapes $\rightarrow$ Digits

### Weights in NNs (Weighted Combination of the input neurons)
* Weights will, on an input into a neuron, will capture a particular pattern/portion within the input (could be all of it)
    * How do we pick up the pattern with a neuron?
* For each of the inputs going into a neuron, we will assign a value/multiplier to the input 
* Goal: Neuron lit up when a certain pattern exist
* $w_1a_1 + w_2a_2 + w_3a_3 + \ldots + w_n a_n$
* Positive weights on the pattern, negative for the surronding (for instance)
    * To ensure any pixel outside is "not lit up"

### Activation Function in NNs
* After getting our linear combination, we will then pass it through some function
* Should be in range $[0, 1]$
    * "Valid Probabilities"
* In the example, we use the sigmoid function to determine how "positive" the region was for the image detection of digits
    * Sigmoid: $ \sigma{(x)} = \frac{1}{1 + e^{-x}}$
    * Very large negative numbers taken to 0, very large positive numbers taken to 1
    * $x = 0 \rightarrow \sigma{(x)} = \frac{1}{2}$
* Other Functions:
    * Hyperbolic Tangent: $f(x) = \frac{2}{1+e^{-2x}} - 1$
    * Rectified Linear Unit (ReLU): $f(x) = \max{(0,x)}$, basically ignore all negative values (mapped to 0)
* These are all nonlinear functions
* Can we have linear/Affine activation functions?
    * It would end up making the Neural Network only have 1 layer. It does not really do anything.
    * Composing multiple linear transformations together is still one large linear transformation
    * Yes, but technically no

### Bias in NNs
* Bias for inactivity
* Essentially a weight/constant that is not multiplied with an input of a neuron

### Parameters in Neural Networks are all weights are biases!
* 1 bias per neuron
* Weights is number of inputs from last layer times number of inputs into each neuron
    * Example in Class: $784 \times 16 + 16 \times 16 + 16 \times 10$
    * For a fully connected layer
* Learning in a NN: Finding the right weights and biases for the model

### Back to Hidden Layers
* We should not treat the layers as a Black Box
    * Instead, we need to understand why the neural networks behave as they do

### Notation for NNs
* $a^{(n)}_x$
    * $n$ corresponds to the number layer you are at (indexing at 0)
    * $x$ is the $x$-th input within the layer (indexing at 0)
* $w_{x,y}$ (for weights)
    * $x$ is the $x$-th neuron in the output
    * $y$ is for the $y$-th input into that $x$-th neuron.
* $b_x$
    * Bias for the $x$-th output neuron
* $W$ is the matrix of weights, rows are the number of neurons in the ouput layer, columns are the number of neurons from the input layer going into that neuron. (Fully connected, it will be $n$)
    * WILL BE DIFFERENT IF NOT A FULLY CONNECTED LAYER
* Activation Functio applied to each row, i.e.
    * $\sigma(\begin{bmatrix}x \\ y \\ z\end{bmatrix}) = \begin{bmatrix}\sigma(x) \\ \sigma(y) \\ \sigma(z)\end{bmatrix}$
* $a^{(1)} = \sigma(Wa^{(0)} + b)$
    * $W$ is $k\times n$, $k$ is for output, $n$ for input
    * $a^{(0)}$ will be $n\times1$
    * $b$ is $k \times 1$
    * $a^{(1)}$ will be $k\times 1$
* This makes coding much easier, can vectorize via Python
* This if for the "Feed-Forward" type of NN

### Returning to Neurons
* "Neurons are a number" is false
* "Neurons are a function" is more apt 

### Universal Approximators
* $f(x) = \sigma(w_1x + b_1) + \sigma(w_2x + b_2) + \ldots$
* We can approximate any continuous function with a linear combination of translated/scaled ReLU functions
    * $\psi \in C([a,b], R)$
* True for other activation functions under mild assumptions

### Universal Approximation Theorem
* One can approximate any continuous functoin at any precision with a single hidden layer MLP (multilayer perceptron) with enough hidden units

### Universal Approximation via NNs
* Single-hidden-layer MLP is a universal approximator
    * Can approximate any Boolean function, classification function, or regression function to arbitrary precision
    * May require infinite neurons in the layer
* Can approximate, but not an exact answer!

### Sufficiency of Architecture
* NN can represent any function provided sufficient capacity
    * Broad and deep enough
* Not all architectures can represent any function
* Example:
    * A network with 8 threshold neurons in the first layer may capture these 8 boundaries
    * Can give you info on which of the trips in the image the input is in, but not where in the strip
    * Even if the 8 first-layer neurons capture these boundaries, they can only place you in one of 25 cells, but not where.
* Regardless of depth, each layer must be sufficiently wide to capture the function