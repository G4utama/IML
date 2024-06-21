def split_data(x, y, train_ratio=0.8, seed=None):
    idx_samples = np.arange(len(x), dtype=np.int32)
    if seed is not None:
        np.random.seed(seed)

    ### HERE YOUR CODE ###
    np.random.shuffle(idx_samples) # Shuffle the idx samples
    train_size = int(np.ceil(x.shape[0]*train_ratio)) # compute the train set size
    idx_train, idx_valid = idx_samples[:train_size], idx_samples[train_size:] # Split the idx into train and validation idx
    ### END CODE ###

    return x[idx_train], y[idx_train], x[idx_valid], y[idx_valid]