def dist_l2(x_i, x):
    '''
    input x_i: input sample ndarray of shape (feat_dim,)
    input x: input samples ndarray of shape (num_samples, feat_dim)
    output d_i: L2 distance between xi and each sample in x, ndarray of shape (N,)
    '''
    ### HERE YOUR CODE ###
    d_i = np.sum((x_i-x)**2, axis = -1)**0.5
    ### END CODE ###
    return d_i