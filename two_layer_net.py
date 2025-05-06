"""
Implements a two-layer Neural Network classifier in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import numpy as np

def hello():
    print('Hello from two_layer_net.py!')


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in the variable 'out'#
    ###########################################################################

    out = x @ w + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    assert out is not None, "You are asked to update out properly"
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    # Note: 'db' is 1D, not 2D, vector.                                       #
    ###########################################################################

    dx = dout @ w.T
    dw = x.T @ dout
    db = np.sum(dout, axis=0).flatten()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    assert dx is not None, "You are asked to update dx properly"
    assert dw is not None, "You are asked to update dw properly"
    assert db is not None, "You are asked to update db properly"
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass. update 'out'                     #
    ###########################################################################

    indicator = x > 0
    out = indicator * x

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    assert out is not None, "You are asked to update out properly"
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    indicator = x > 0
    dx = np.multiply(dout, indicator)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    assert dx is not None, "You are asked to update dx properly"
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)

    loss = 0.0
    N = x.shape[0]
    dx = probs.copy()
    ###########################################################################
    # TODO: compute the softmax loss and store it to 'loss'.                  #
    # Also, you are required to compute the gradient to 'dx'.                 #
    # Hint for both: Please do not forget to consider scale of the gradient.  #
    # Hint for loss: Check page 47 of Lecture 11 for the Cross Entropy loss.  #
    # Hint for loss: Please check what 'log_probs' is.                        #
    # Hint for dx: Note that we already copied probs into dx.                 #
    ###########################################################################
    
    oneHotY = np.zeros((y.size, x.shape[1]), dtype=int)
    oneHotY[np.arange(y.size), y] = 1
    loss = -np.sum(oneHotY * log_probs) / N
    dx = (dx - oneHotY) / N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    assert loss != 0, "You are asked to update loss properly"
    assert np.any(probs - dx != 0), "You are required to update the dx value"
    return loss, dx


class TwoLayerNet:
    """
    A fully-connected neural network with softmax loss that uses a modular
    layer design.

    We assume an input dimension of D, a hidden dimension of H,
    and perform classification over C classes.
    The architecture should be fc - relu - fc - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100,
                 num_classes=10, weight_scale=1e-3):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params. Use keys 'W1' and 'b1' for the weights and       #
        # biases of the first fully-connected layer, and keys 'W2' and 'b2' for    #
        # the weights and biases of the output affine layer.                       #
        ############################################################################

        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params['b1'] = np.zeros((hidden_dim,))
        self.params['b2'] = np.zeros((num_classes,))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        assert 'W1' in self.params, "You are asked to add W1"
        assert 'W2' in self.params, "You are asked to add W2"
        assert 'b1' in self.params, "You are asked to add b1"
        assert 'b2' in self.params, "You are asked to add b2"

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Inputs:
        - X: Array of input data of shape (N, d_in)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        # Please do not reimplement fc_forward and relu_forward from scratch.      #
        # HINT: Rember to save cache! (See the returns of each forward function    #
        # and the inputs of each backwards function)                               #
        ############################################################################

        out, cahce1 = fc_forward(X, self.params['W1'], self.params['b1'])
        out, cahce2 = relu_forward(out)
        scores, cache3 = fc_forward(out, self.params['W2'], self.params['b2'])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        assert scores is not None, "You are asked to update scores properly"

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Please do not reimplement softmax_loss, fc_backward, and #
        # relu_backward from scratch.                                              #
        ############################################################################

        loss, derivative = softmax_loss(scores, y)
        derivative, grads['W2'], grads['b2'] = fc_backward(derivative, cache3)
        derivative = relu_backward(derivative, cahce2)
        _, grads['W1'], grads['b1'] = fc_backward(derivative, cahce1)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        assert 'W1' in grads, "You are asked to add W1"
        assert 'W2' in grads, "You are asked to add W2"
        assert 'b1' in grads, "You are asked to add b1"
        assert 'b2' in grads, "You are asked to add b2"

        return loss, grads
