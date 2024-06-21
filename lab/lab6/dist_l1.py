def dist_l1(x_i, x):
    '''
    input x_i: input sample ndarray of shape (feat_dim,)
    input x: input samples ndarray of shape (num_samples, feat_dim)
    output d_i: L1 distance between xi and each sample in x, ndarray of shape (N,)
    '''
    ### HERE YOUR CODE ###
    d_i = np.sum(np.abs(x_i-x), axis = -1)
    ### END CODE ###
    return d_i