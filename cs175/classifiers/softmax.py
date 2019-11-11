import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. (3073, 10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (500, 3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C. (500, 1)
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
    
    num_train = X.shape[0]  
    num_classes = W.shape[1]

    # Loss   
    term1 = 0
    for i in range(num_train):
        term2 = 0
        for j in range(num_classes):
            Wt = np.transpose(W[:,j])
            score = np.dot(Wt, X[i,:])
            term2 += np.exp(score)
        term1 += (-np.log((np.exp(np.dot(np.transpose(W[:,y[i]]), X[i,:])))/term2))
        
    loss = (1/num_train)*term1
    
    l2_norm = 0
    for c in range(num_classes):
        l2_norm += np.linalg.norm(W[:,c])**2
    loss += reg*l2_norm
    
    # dW
    term3 = 0
    for i in range(num_train):
        hx = np.zeros(num_classes)
        x2d = np.atleast_2d(X[i,:])    #(1,3073)
        for j in range(num_classes):
            denom = 0
            for k in range(num_classes):
                Wtk = np.transpose(W[:,k])
                score_k = np.dot(Wtk, X[i,:])
                denom += np.exp(score_k)
            Wtj = np.transpose(W[:,j])    #(3073,1)
            score_j = np.dot(Wtj, X[i,:])
            hx[j] = np.exp(score_j)/denom
        y_hot = np.zeros(num_classes)
        y_hot[y[i]] = 1
        y2d = np.atleast_2d(hx-y_hot) #(1,10)
        term3 += np.dot(x2d.T, y2d)
    
    dW = (term3/num_train) + 2*reg*W
                               
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
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # W: (D, C) (3073, 10)
    # X: (N, D) (500, 3073)
    # Y: (N, 1) (500, 1)
    
    # Loss
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    scores = np.dot(X, W)
    max_score = np.max(scores)
    scores -= max_score 
    exp_score = np.exp(scores) #(N, C) (500, 10)
    
    sum_score = np.sum(exp_score, axis=1)
    sum_score = np.atleast_2d(sum_score).T #(N, 1) (500, 1)
    
    yhat = exp_score / sum_score #(N, C) (500, 10)
   
    one_hot = np.zeros((num_train,num_classes)) #(N, C) (500, 10)
    indexs = np.arange(num_train) #(N) (500)
    one_hot[indexs, y] = 1
    
    loss = -one_hot*np.log(yhat)
    loss = np.mean(np.sum(loss,axis=1))
    loss += np.linalg.norm(W)**2
    # dW
    dW = (np.dot(X.T,(yhat-one_hot))/num_train) + 2*reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW