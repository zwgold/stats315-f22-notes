# STATS 315: Lecture 4 (Logistic Regression as a Neural Network II + DL Frameworks)

## Topics 
* Gradient descent
* Derivatives with a computational graph
* Logistic regression with gradient descent
* TensorFlow, Keras, Google Colab
* JiTTs suck, as usual

## Binary Classification

### Recall
* Used logistic regression
* Input: Image X (unroll pixels into feature vector)
* Output: Label Y that takes value 1 or 0
    * "Is or isn't a thing"
    * "Yes or No"
    * "True or False"

### Convert images to vectors
* $x \in \mathbb{R}^{n_x}, y\in\{0,1\}$

### Notation
* $(x,y), x \in \mathbb{R}^{n_x}, y\in\{0,1\}$
* $m$ training examples: $\{(x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)})  \}$
* Training examples: $m_{train} = m$
    * Testing examples: $m_{test}$
* $x \in \mathbb{R}^{n_x \times m}$
    * Shape is $(n_x,m)$
    * Stacked column wise for data
    * $m$ columns for number of datapoints
    * $n_x$ rows for number of features in each datapoint
* $y \in \mathbb{R}^{1\times m}$
    * Shape is $(1,m)$

### The process
* Given $x$, want to predict $\^{y} = P(y = 1 | x)$
* $x\in\mathbb{R}^{n_x}, 0 \le \^{y} \le 1$
* Parameters:
    * $w\in\mathbb{R}^{n_x},b\in\mathbb{R}$
* Output $\^{y} = \sigma{(\sum^{n_x}_{j=1}w_jx_j + b)} = \sigma{(w^\intercal x + b)}$
* $\sigma{(z)} = \frac{1}{1 + e^{-z}}$
* Large Positive $z$ means we get closer to 1, Small/Large Negative $z$ means we get closer to 0

### Scalar and Matrix Form
* $d$-dimensional input
* $\^{y}^{(i)} = \sigma{(\sum^{n_x}_{j=1}w_jx^{(i)}_j + b)} = \sigma{(w^\intercal x^{(i)} + b)}$
    * $\sigma{(z^{(i)})} = \frac{1}{1 + e^{-z^{(i)}}}$
* Given our training data, want $\^{y}^{(i)} \approx y^{(i)}$
* Loss Function/Error Function: $L(\^{y}, y) = -(y\log{\^{y}^{(i)}} + (1-y)\log{(1-\^{y}^{(i)})})$
    * Convex function
* Cost Function:
    * $J(w, b) = \frac{1}{m}\sigma{L(\^{y}^{(i)}, y^{(i)})}$
    * Full form from last lecture

### Cost Function / Loss Function
* Loss function measures how good the prediction is relative to the true label
* Loss function / error function, see above
* Training: Make the cost function as small as possible
* Cost function: Cost for parameters for the **whole** training set
* Desired to have convex loss functions!

## Gradient Descent

### Recall the above Loss / Cost Function requatiosn
* $\^{y}^{(i)} = \sigma{(\sum^{n_x}_{j=1}w_jx^{(i)}_j+b)} = \sigma{(w^\intercal \bf{x^{(i)}} + b )}$ where $\sigma{(z^{(i)})} = \frac{1}{1+ e^{-z^{(i)}}}$
* $J(w, b) = \frac{1}{m}\sum^{m}_{i=1}L(\^y^{(i)}, y^{(i)}) = -\frac{1}{m}\sum^{m}_{i=1}(y^{(i)}*\log{\^{y}^{(i)}} + (1-y^{(i)})*\log{(1-\^{y}^{(i)})})$
* Find $w,b$ s.t. we minimize our cost function

### Definition
* Gradient Descent Process:
    1. Initialize
    2. Take a step in the steepest downhill direction at each iteration
    3. Repeat until the algorithm converges
* Initalize $W$ to some number
    * Either from a uniform sample $[-1,1]$ or $N(0,1)$
* Repeat
    * $ w \leftarrow w - \alpha \frac{dJ(w)}{dw}$
        * $\alpha$ is the learning rate
        * $\frac{dJ(w)}{dw}$ value of derivative of cost function after substituting $w$ in
    * Repeat until convergence (within some threshold of changes)
* When we work with a multivariate function: $J(w, b)$
    * $w \leftarrow w - \alpha\frac{\partial J(w,b)}{\partial w}$
    * $b \leftarrow b - \alpha\frac{\partial J(w,b)}{\partial b}$
* Learning Rate: Control how big a step we take at each iteration
* Derivate: Slope of the function; direction goes downhill
* Code: often use "dw" to present derivative "dJ(w)/dw"
    * Amount you want to update for w
* FALSE: Convex functions always have multiple local optima
    * Only 1 local/global

### Intuition on Derivatives
* Example Function:  $f(x) = 3x$
* Slope/Derivative: If I nudge $x$ by some small amount, I expect it to change/move by 3 times as large as the nudge
* Refer to Calc 1, 2, 3, 4 for this stuff

### Computational Graph
* Computations of a NN are organized as such:
    * Forward pass / Forward propagation step, where we compute the output of a NN
    * Followed by a backward pass / Back propagation, which we use to compute gradients of compute derivatives
* Computation graph explains why it is organized as such

### Right-to-left Pass (Chain Rule)
* Example: $J(a,b,c) = 3(a + bc)$
    * $u = bc$
    * $v = a + u$
    * $J = 3v$
* $a \rightarrow v \rightarrow f$
* Backprop: $\frac{dJ}{da}$