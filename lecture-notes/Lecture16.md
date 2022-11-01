# STATS 315: Lecture 16 (Fundamentals of ML IV + CNN I)

## Logistics
* 1 % point on onverall grade for final evaluation

### JiTTs
* Hyperparameters --> values that affect output of NN aside from inputs and learned parameters
    * No weights or biases
    * Might be scalars to multiply activation functions
* K-Fold Cross Validation vs Standard Cross Validation
    * Lower variance (K Fold)
    * Faster (Standard)
    * Lower Bias assuming same validation sets (Standard)
* Vectorization --> Faster, multiple things at a time, not sequential like a for loop, GPU or TPU usage 
    * Better than a for loop
* Overfitting
    * Validation loss will start to increase at end of training
    * Test-set loss substantially worse than validation loss, and you used validation set to choose among 100 hyperparameter configs
        * If we had only fit 1 model --> won't be a result of overfitting
    * Train Loss will plateau and flatten
* Depth will have a greater impact on the ability to approximate a function?

## Convolutional NNs and CV