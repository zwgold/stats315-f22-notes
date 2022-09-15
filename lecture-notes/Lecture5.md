# STATS 315: Lecture 5 (Computational Graphs + TensorFlow)

## Topics 
* Computational Graphs
* TensoFlow
* DL Frameworks

## Computation Graph

### Recall
* Organized by:
    * Forward pass/Forward propagation step, where we compute the output of a NN
    * Backward pass/Backward propagation step, where we compute gradients/derivatives
* Graph explains why it is organized this way

### Left-to-Right Pass
* Assume $J(a,b,c) = 3(a +bc)$
* Three Steps: $u = bc, v= a+ u, J = 3v$
* $\frac{dJ}{dv} = ?$
* Backprop:d
* $\frac{dJ}{da} = \frac{dJ}{dv}\frac{dv}{da}$
    * Propagate the change in a to v to J
* From example in class:
    * $a = 5, b = 3, c = 2$
    * $u = 6, v = 11, J = 33$
    * $\frac{\partial J}{\partial v} = 3$
    * $\frac{\partial J}{\partial a} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial a} = 3 * 1 = 3$
    * $\frac{\partial J}{\partial b} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial u}\frac{\partial u}{\partial b}$
* Notation Convention:
    * If I wanted to know what the derivative of some final output variable (w.r.t some other variable)
    * $dJdvar$
        * Example: $\frac{\partial J}{\partial a} = dJda$
    * Often: Cost Function
* Just do the basic gradient calculations with the equations and we'll be fine

### Logistic Regression with GD
* $z = \sum_j^{n_x}w_jw_i + b = w^\intercal x + b$
* $\^{y} = a = \sigma{(z)}$
* $L(a,y) = -(y\log{(a)} + (1-y)\log{(1-a)})$
* Computation Graph of logistic regression
* Derivatives:
    * $\frac{\partial L}{\partial z} = (-\frac{y}{a} + \frac{1-y}{1-a})*(a(1-a))$
    * $\frac{\partial L}{\partial w_1} = above*x_1$
    * $\frac{\partial L}{\partial w_2} = above*x_2$
    * $\frac{\partial L}{\partial b} = above * 1$
* $m$ examples
    * $J(w,b) = \frac{1}{m}\sum_{i=1}^m L(a^{(i)}, y^{(i)})$
    * $a^{(i)} = \^{y}^{(i)} = \sigma{(z^{(i)})} = \sigma{(w^\intercal x^{(i)}+ b)}$
    * Previously done with only 1 example
    * $\frac{\partial}{\partial w_1}J(w,b) = \frac{1}{m}\sum_{i=1}^m \frac{\partial}{\partial w_1}L(a^{(i)}, y^{(i)})$
* Full alg: (Image will be added)

## Deep Learning Frameworks
* We can avoid reimplementing the same structure over and over again
* Choosing DL frameworks:
    * Ease of programming (development and deployment)
    * Running speed
    * Truly open (open source, good governance)
* See Code examples in slides

### TensorFlow
* Can leverage it to get a Computation Graph
* See Notebook