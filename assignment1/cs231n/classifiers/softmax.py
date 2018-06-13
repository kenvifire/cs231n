import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_class = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = np.exp(X[i].dot(W))
    sum = scores.sum()
    pk = scores / sum
    margin = -np.log(pk)
    loss += margin[y[i]]
    for j in xrange(num_class):
      df = 0.0
      if j == y[i]:
        df = pk[j] - 1
      else:
        df = pk[j]
      dW[:,j] += df * X[i]

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.exp(np.dot(X, W))
  probs = scores / np.sum(scores, axis=1, keepdims=True)
  log_probs = -np.log(probs[range(num_train), y])
  loss = np.sum(log_probs) / num_train + 0.5 * reg * np.sum(W*W)

  dscores = probs
  dscores[range(num_train), y] -= 1
  dscores /= num_train
  dW = np.dot(X.T, dscores) + reg * W




  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

