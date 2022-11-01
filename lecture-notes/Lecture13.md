# STATS 315: Lecture 13 (Fundamentals of ML1)

## JiTT: Forward and Backward Props + Weights
* Forward --> Calculate outputs
* Backward --> Get gradients to update model parameters
* Computational Graph shows both of these
* You can perform backprop without a NN computational graph --> FALSE
* Weights to zero --> JUST FOR DEEP NNs
    * Logistic Regression --> Cross Entropy Loss, Gradient Descent
* $L(a,y)=-y\log{a} - (1-y)\log{(1-a)}$
    * Convex, find the minimum value of the loss function
* $a=\sigma{(\sum w_ix_i + b)}$

## Fundamentals of ML
* Covered in the Colabs in class