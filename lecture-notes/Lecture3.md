# STATS 315: Lecture 3 (Logistic Regression as a NN)

## Topics
* Log Regression
    * How it is used in binary classification
    * How it functions as a NN
    * 1 input, 1 output, 2 nodes/neurons
* Loss functions
* GD
* Derivatives with a computational graph

## From JiTTs:
* Lectures highlight important concepts + methods that are essential for understanding + implementing DL algs
* Readings offer more extensive descriptions, examples, and details
* Quizzes are on **basic** concepts
* Labs + HWs support learning understanding of algorithmic implementations through examples and exercises (only HWs are graded)
* Projects will have you find you own dataset and implement your own DL Algs

## Syllabus Updates/Reminders
* Frop 4 lowest JiTTs
* Use JiTTs to guide your reading
* Serves as a "check-in" before class, should not take more than 10 minutes to complete. Graded for EFFORT and NOT ACCURACY.

## Project Info
* Use kaggle
* Final Project guidelines are updated
* HWs will teach you to load databases and perform analysis

## Training NNs
* Idea 1: Process entire training set, no explicit for loop to go through entire set
* Idea 2: Why the computations in learning a NN can be organized in forward propagation and a separate backward propagation
* These ideas will be conveyed through **logistic regression**, which should make it easier to understand

## Binary Classification
* Input: Image X (unroll pixels into feature vector)
Output: Label Y, takes on 2 values
    * 0 or 1
    * In our example, 1 is a cat, 0 is non-c
    
### Image to Input Feature Vector
* Map the pixels and values to some vector
* $x^{(i)} = [56, -3.2, 69, 15689]$ (vanilla case)
    * ($i$-th training value)
    * Example could be background, corner, contour, patch
* What do we do with the iage?
    * If we have a $5 \times 4$ image, we have $3 
    \times 5 \times 4$ colors/numbers
    * $3$ corresponds to the number of color channels
    * $x = \begin{bmatrix} -255\\231\\ \vdots\\ 255\\ 135\\ \vdots 255 \\ 134\\ \vdots \end{bmatrix}$ (red, then green, then blue)
    * In our example, we would have 60 entries (that's a lot)
* $n_x$ is the dimensionality of the input vector
* Goal: Take in the input ($x$) and produce a label ($y$)

### Notation
* Data point: $(x, y)$
    * $x$ is the input vector, $y$ is the label
    * $x \in \mathbb{R}^n$; the whole picture
    * $y \in \{0 , 1\}$
* Dataset can have $m$ examples/datapoints
    * $m = m_{train}$
    * $m_{test}$ is the number of test examples
* $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(m)}, y^{(m)}) \}$
    * We can organize this into 2 matrices
* $X = \begin{bmatrix} x^{(1)} & x^{(2)} & \ldots & x^{(m)} \end{bmatrix}$ 
    * Number of rows is $n_x$-dim
    * Number of columns is $m$-dim
* $Y = \begin{bmatrix} y^{(1)} & y^{(2)} & \ldots & y^{(m)} \end{bmatrix} \in \{0,1\}^m$ 
    * $1 \times m$ vector

## Logistic Regression

### Scalar and Matrix Representaions
* $\^{y} = \sigma{(\sum^{n_x}_{j=1} w_jx_j + b)}$
    * y-hat is our prediction
    * We use the sigmoid function
* Sigmoid: $\sigma{(z)} = \frac{1}{1 + e^{-z}}$
* Goal: Given dataset, we want $\^{y}^{(i)} \approx y^{(i)}$ from logistic regression
    * Want our prediction to match the true label from the dataset 
* We want to find values of $w, w_{n_x}, b$ 

### Loss Functions
* Function (also called error function) that measures how good/bad our prediction is versus our actual data/the truth
    * More error for a greater difference
* $L(\^{y}, y) = \frac{1}{2}(\^{y} - y)^2$ 
    * Used for linear regression
    * Non-convex (i.e. there is some "wiggly" nature, values that are not as deep as the overall deepest/tallest part)
* Convex Function Example: $L(\^{y}, y) = -[y*\log{\^{y}} + (1-y)*\log{(1-\^{y})}]$
    * If $y = 1$, then $L(\^{y}, y) = -\log{\^{y}}$
         * How do I minimize the loss?
            * If I want $-\log{\^{y}}$ to be small, we want $\^{y}$ very large
    * If $y = 0$, then $L(\^{y}, y) = -\log{(1 - \^{y})}$
         * How do I minimize the loss?
            * If I want $-\log{(1 - \^{y})}$ to be small, we want $\^{y}$ very small

### Cost Function
* Cost Function: 
    * $J(w, b) = \frac{1}{m}\sum^{m}_{i=1}L(\^y^{(i)}, y^{(i)}) = -\frac{1}{m}\sum^{m}_{i=1}(y^{(i)}*\log{\^{y}^{(i)}} + (1-y^{(i)})*\log{(1-\^{y}^{(i)})})$
        * For whole dataset
* Comparing loss function for a single training example / datapoint