from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    ######################)#######################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(X.shape[0]):
      score = X[i].dot(W)
      score -= np.max(score)
      loss  += -np.log(np.exp(score[y[i]])/np.sum(np.exp(score)))
      # loss += -score[y[i]] + np.log(np.sum(np.exp(score)))
      for j in range(W.shape[1]):
        #if the class is for the vector in consideration the formula is different
        if j ==y[i]:
          p= np.exp(score[j])/np.sum(np.exp(score))-1
          dW[:,j]+= p*X[i]
        else:
          p = np.exp(score[j])/np.sum(np.exp(score))
          dW[:,j] += X[i]*p
    loss /= X.shape[0]
    loss += reg * np.sum(W * W)
    dW /= X.shape[0]
    dW = dW + 2*reg*(W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    scores = X.dot(W)
    #to avoid numerical inconsistency
    scores = np.subtract(scores,np.amax(scores,axis=1).reshape(-1,1))
    #remove the required elements form the correct class
    score_y_i = scores[np.arange(X.shape[0]),y]
    #compute the loss
    loss = np.sum(-np.log(np.exp(score_y_i)/np.sum(np.exp(scores),axis=1)))

    scores_all = np.exp(scores)
    score_sum = np.sum(np.exp(scores),axis=1)
    dW = scores_all/score_sum.reshape(-1,1)
    dW[np.arange(X.shape[0]),y]-=1
    dW = X.T.dot(dW)
    dW /= X.shape[0]
    dW += reg*W
    loss /= X.shape[0]
    loss += reg*np.sum(W*W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
