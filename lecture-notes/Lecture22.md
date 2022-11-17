# STATS 315: Lecture 22 (RNN III)

## JiTTs
* RNN Appropriate for:
    * Time data
    * Text to Speech / Speech
    * Predictors for future (customer payments)
    * If a short video contains something
    * Song classification
    * Detection of anomalies
* Can combine CNN layers and RNN layers

## Attention

### Neurons with Recurrence
* $\hat{y}_t=f(x_t,h_{t-1})$
* Input vector
* Output vector
* Some latent/memory state that is how much info we want to keep over
* For Memory State:
    * $h_t = f_W(x_t,h_{t-1})$
    * Same function and set of parameters USED AT EVERY TIME STEP
* RNNs have a state that is updated at each time step as a sequence is processed

### LSTM
* Gated LSTM (Long Short Term Memory) controls info flow
    * Forget, Store, Update, Output
    * Tracks information throughout many timesteps
* Maintain a separate cell state from what is outputted
* Use gates to control flow of info
    * Forget gate gets rid of irrelevant info
    * Selectively update cell state
    * Output gate returns a filtered version of the cell state
* Backprop through time with partially uninterrupted gradient flow

### Limitations of Recurrent Models
* Encoding Bottleneck
    * Cannot afford to make memory state very large for longer sequences (even if we want flexibility)
* Slow, no parallelization
    * Need to do things in order, one by one; precludes parallelization
* NOT long memory
    * Designed to be long range memory, but run into vanishing and exploding gradients
    * RNN cannot handle long range in a smooth manner

### Goal of Sequence Modeling
* Desired Capabilities
    * Continuous Stream
    * Parallelization
    * Long Memory
* Can we eliminate the need for recurrence? 
* Idea 1: Feed everything into dense network
    * No Recurrence (Good!)
    * Not scalable, no order, no long memory (Bad!)
    * Use input, move into feature vectore, then output
        * Some magic in the feature vector
    * **What is important for us to attend to?**
* **Attention is all you need**

### Intuition of Self-Attention
* Attend to the most imporatnt parts of an input
1. Identify Parts to attend
    * Search Problem?
2. Extract features with high attention

### Learning Self-Attention with Neural Networks
* Goal: Identify and attend to most importatn features in input
1. Encode position information
    * Data fed in all at once! Need to encode position information to understand order.
    * Get the position information / encode it, add to the original word embeddings
    * Undergoes masking
2. Extract query, key, value for search
    * Using a mask (linear layer) to transform the positional embedding
    * 3 separate masks for Query (Q), Key (K), and Value (V)
3. Compute attention weighting
    * Mask that is the same size as the input, but tells us what the most important parts are
    * Attention Score: Compute pairwise similarities between each query and key
    * How do we compute simiarlity between two sets of features?
    * $Q \cdot K^\intercal \rightarrow \frac{Q\cdot K^\intercal}{scaling}$
    * Apply to activation function (Softmax)
* Attention weighting: Where to attend to!
    * How similar is the key to the query?
        * Do for each pair 
    * Key is vertical line, Query is horizontal
4. Extract features with high attention
    * Self-attend to extract features
    * Use attention weighting matrix, mutliply with Value to get output
    * $softmax(\frac{Q\cdot K^\intercal}{scaling})\cdot V = A(Q,K,V)$
        * With $V$ is dot product
* These operations --> self-attention head that can plug into a larger network. Each head attends to a different part of the input

### RNN Summary
1. RNNs are well suited for sequence modeling tasks
2. Model sequences via a recurrence relation
3. Training RNNs with backpropagation through time
4. Models for music generation, classification, machine translation
5. Self-attention to model sequence without recurrence