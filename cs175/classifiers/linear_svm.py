import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3073,10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (500,3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C. (500,)
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero (3073,10)
    
    # compute the loss and the gradient
    num_classes = W.shape[1] # 10
    num_train = X.shape[0]   # 500
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i,:].dot(W) # (C,1)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if j == y[i]:
                dWk = 0.0
                for k in xrange(num_classes):
                    if k == y[i]:
                        continue
                    marginK = scores[k] - correct_class_score + 1
                    if marginK > 0:
                        dWk += 1
                dW[:,j] -= dWk*X[i,:]
            else:
                if margin > 0:
                    loss += margin
                    dW[:,j] += X[i,:]
                    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2*reg*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    scores = np.dot(X, W) #(N,C) (500,10)
    correct_scores = np.zeros(num_train)
    indexs = np.arange(num_train)
    correct_scores = scores[indexs, y]
    margins = np.maximum(0, scores - np.matrix(correct_scores).T + 1)
    margins[indexs,y] = 0
    
    loss = np.mean(np.sum(margins, axis=1))
    loss += 0.5 * reg * np.sum(W*W)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    margins_2 = margins
    margins_2[margins > 0] = 1
    sum_margins = np.sum(margins_2, axis=1)
    margins_2[indexs,y] = -sum_margins.T
    dW = np.dot(X.T, margins_2)
    
    dW /= num_train
    dW += reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
