class KNNModel(object):
    def __init__(self, x, y, k=1, num_classes=10):
        self.k = k
        self.x = x
        self.y = y
        self.num_classes = num_classes

    def predict(self, x, dist_func, get_freq=False):

        def get_k_closest_points(x_i, x_list):
            '''
            input x_i: input sample ndarray of shape (feat_dim,)
            input x_list: input samples ndarray of shape (num_samples, feat_dim)
            output idx_k: indices of the kNN, ndarray of shape (k,)
            '''
            ### HERE YOU CODE ###
            d_list = dist_func(x_i, x_list)
            idx_k = np.argsort(d_list)[:self.k]
            ### END CODE ###

            return idx_k

        # Compute the distance between x and self.x using dist_func
        dist_matrix_k = np.zeros([x.shape[0], self.k], dtype=np.int32)
        for i, x_i in enumerate(x):
            dist_matrix_k[i, :] = get_k_closest_points(x_i, self.x)

        # Voting
        y_pred_k = self.y[dist_matrix_k]

        # Get the most frequent class and also all the voting frequency
        if get_freq:
            y_pred_freq = np.zeros([len(y_pred_k), self.num_classes], dtype=np.float32)
            for cl in range(self.num_classes):
                idx = np.where(y_pred_k == cl)
                idx_row = np.array(list(set(idx[0].tolist())), dtype=np.int32)
                y_pred_freq[idx_row, cl] = (y_pred_k[idx_row] == cl).sum(axis=-1)
            y_pred_freq = y_pred_freq / self.k
            return y_pred_freq.argmax(axis=-1), y_pred_freq
        else:

            ### HERE YOUR CODE ###
            mode = stats.mode(y_pred_k.T)[0]
            ### END CODE ###

            return mode