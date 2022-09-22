# STATS 315: Lecture 7 (Vectorization and Linear Algebra Bootcamp I)

## Vectorization?

### What is Vectorization?
* Vectorization: Removing explicit for loops in code with linear algebra
* Python: For loops are flow, compared to the matrix operations
* Computations take a long time with loops
* $z = \sum_{i=1}^d w_ix_i + b$
    * $\^{y} = \sigma(z)$
    * $w = \begin{bmatrix}w_1 \\ w_2 \\ \vdots \\ w_{n_x} \end{bmatrix}$
        * $w \in \mathbb{R}^{n_x}$
    * $\vec{x} = \begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_{n_x} \end{bmatrix}$
        * $\vec{x} \in \mathbb{R}^{n_x}$
* Non-vectorized:
```
z = 0
for i in len(nx):
    z += w[i] * x[i]
z += b
```
* Vectorized:
```
z = np.dot(w, x) + b
```
* GPU / CPU has parallelization instructions
    * SIMD --> "single instruction multiple data"
* Built in functions of numpy and tensorflow (and other DL frameworks) take better advantage of parallelism to speed up calculations
    * True both on CPUs and GPUs
        * GPUs extremely good at SIMD
            * GPU class crying right now
        * CPUs not too bad either
* Rule of thumb: Avoid for loops if possible

## Linear Algebra

### Mathematical Objects in Linear Algebra
* Scalar
    * Just a number
* Vector
    * Ordered array of numbers, row or column
    * Has a single index to point to a specific value within vector
* Matrix
    * Ordered 2D array of numbers, 2 indices
    * First one points to row, second one to column
* Tensor
    * Array of numbers, arranged on a regular grid, with a variable number of axes
    * Tensor has 3 indices, first one points to row, second to column, third to axes
    * 1D --> Vector, 2D --> Matrix, 3D --> higher dimensional

### Computational Rules
* Matrix-Scalar Operations:
    * Add/Sub/Mult/Div by element 
* Matrix-Vector Operations
    * Matrix by vector multiplication can be thought of as multiplying each row of the matrix with the column of the vector
    * Output is a vector with the same number of rows as the matrix (but in column form)
* Matrix-Matrix
    * Add and subtract by element
    * Multiply by splitting second matrix into column vectors, multiple first matrix separately by each of these.
    * Put results in a new matrix, DO NOT add them up
    * NOT COMMUTATIVE
    * Are: Associative, Distributive
    * Identitiy Matrix: All zeros except for main diagonal which has 1's
* Transpose
    * Mirror image of the Matrix, rows --> columns and columns --> rows
    * m by n becomes n by m
    * Aij element of A is equal to the Aji (transpose) element

## Back to Vectorization

### Examples
* Matrix A that is $m \times d$ and vector v that is $d \times 1$
* $u = Av$
* $u_i = \sum_j A_{ij}v{j}$
```
u = np.zeros((n, 1))
for i ... :
    for j ... :
        u[i] += A[i][j]*v[j]
```
* Can vectorize easily
* Say you need to apply the exponential operation on every element of a matrix/vector
```
u = np.exp(n)
u = tf.exp(n)
```

### Logistic Regression Derivatives
![Logistic Regression](/lecture-notes/images/lecture-7-log-reg-1.JPG)
![Logistic Regression 2](/lecture-notes/images/lecture-7-log-reg-2.JPG)