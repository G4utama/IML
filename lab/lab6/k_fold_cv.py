def k_fold_cv(x, y, k=5, seed=None):
    '''
    input x: input samples ndarray of shape (num_samples, feat_dim)
    input y: labels ndarray of shape (num_samples)
    input k: number of folds
    input seed: seed for random shuffle
    '''
    idx_samples = np.arange(len(x), dtype=np.int32)
    if seed is not None:
        np.random.seed(seed)
    
    ### HERE YOUR CODE ###
    np.random.shuffle(idx_samples) # Shuffle the samples indices
    idx_sample_folds = np.split(idx_samples, k) # Split the idx samples into k-folds
    ### END CODE ###

    x_train_folds, y_train_folds = [], []
    x_valid_folds, y_valid_folds = [], []
    for idx_k in range(k):
        idx_train, idx_valid = [], []
        for idx_fold in range(k):
            fold = idx_sample_folds[idx_fold]
            if idx_k == idx_fold:
                idx_valid += [fold]
            else:
                idx_train += [fold]

    ### HERE YOUR CODE ###
        x_train_folds += [np.concatenate([x[fold] for fold in idx_train], axis=0)]
        y_train_folds += [np.concatenate([y[fold] for fold in idx_train], axis=0)]
        x_valid_folds += [np.concatenate([x[fold] for fold in idx_valid], axis=0)]
        y_valid_folds += [np.concatenate([y[fold] for fold in idx_valid], axis=0)]
    ### END CODE ###

    return x_train_folds, y_train_folds, x_valid_folds, y_valid_folds