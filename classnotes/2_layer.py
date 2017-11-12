
"""2 layer neural network
First gate: sigmoid function
Second gate: vector dot product

It should only take ~ 20 lines of code using numpy. What a nightmare to use JavaScript for doing matrix computation.
"""
import numpy as np
from numpy.random import randn

"""Glossary
:param N: Number of training examples.
:param D_in: Dimension of the input vector.
:param D_out: Dimension of the output vector.
:param H: Dimension of the hidden layer activation vector.
:param w1: The first weight matrix going from input layer to hidden layer.
:param w2: The second weight matrix going from hidden layer to output layer.
"""
N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

# We will take 2000 iterations to do gradient descent
for step in range(2000):
    # This shit right here is mindblowing, numpy is awesome.
    h = 1 / (1 + np.exp(-x.dot(w1))) # Forward prop from input layer to hidden layer (N x H)
    y_pred = h.dot(w2) # Forward prop from hidden layer to output layer
    loss = np.square(y_pred - y).sum()
    print(step, loss)

    grad_y_pred = 2.0 * (y_pred - y) # Deriative of np.square (N x D_out)
    grad_w2 = h.T.dot(grad_y_pred) # Gradient of w2 w.r.t loss (H x N)(N x D_out) => (H x D_out)
    grad_h = grad_y_pred.dot(w2.T) # Gradient of h w.r.t loss (N x D_out)(D_out x H) => (N x H)

    grad_w1 = x.T.dot(grad_h * h * (1 - h)) # Gradient of w1 w.r.t loss (using chain rule)

    # Now perform gradient descent
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2
