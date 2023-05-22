# Gradient

## Definitions

### Gradients
In vector calculus, the gradient of a scalar-valued differentiable function f of several variables
is the vector field whose value at a point p is the vector whose components are the partial derivatives of f at p.

Intuitively, you can consider gradient as an indicator of the fastest increase or decrease direction at a point.
   
Computationally, the gradient is a vector containing all partial derivatives at a point.

### Finite Difference

* In simple terms, finite difference is a method used to estimate the derivative (rate of change) of a function by calculating the difference between function values at neighboring points.
* It approximates the derivative by dividing the change in the function values by the corresponding change in the independent variable (usually denoted as x).
* For example, given a set of function values y at different points, you can calculate the finite difference by taking the difference between consecutive y values and dividing it by the corresponding difference in x values.


!!!  The gradient in the finite difference or numpy gradient function deals with discrete data points, whereas the gradient of a function in 3D deals with continuous functions in multi-dimensional spaces.

## Syntax
```python
numpy.gradient(f[, *varargs[, axis=None[, edge_order=1]]])
```

The output of numpy.gradient() function is a list of ndarrays (or a single ndarray if there is only one dimension) corresponding to the derivatives of input f with respect to each dimension. Each derivative has the same shape as input f.