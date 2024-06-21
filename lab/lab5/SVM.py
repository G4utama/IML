##################################################
# SVM Class
##################################################

class SVM(object):
    def __init__(self, theta_dim=3, seed=1234):
        np.random.seed(seed)
        self.theta = np.random.normal(loc=0.01, size=[theta_dim, 1]) # shape [theta_dim, 1]
        self.theta[-1] = 0 # Zero initialization for the bias

    def forward(self, x):
        '''
        input x: ndarray of shape [N, theta_dim]
        output h: ndarray of shape [N, 1]
        '''
        h = None
        ##############################
        # INSERT YOUR CODE HERE
        ##############################
        h = np.dot(x, self.theta)
        ##############################
        # END OF YOUR CODE
        ##############################
        return h

    def loss(self, h, y, C):
        '''
        input h: ndarray of shape [N, 1]
        input y: ndarray of shape [N, 1]
        output j: scalar
        '''
        j = None
        ##############################
        # INSERT YOUR CODE HERE
        ##############################
        cost1 = y * np.maximum(1-h,0)
        cost2= (1-y) * np.maximum(1+h,0)
        B = np.sum(self.theta**2) / 2
        A = np.sum(cost1 + cost2)
        j = C*A + B
        ##############################
        # END OF YOUR CODE
        ##############################
        return j

    def fit(self, x, y, x_val, y_val, C=1, lr=1e-3, iterations=100, print_every=10, batch_size=32, seed=1234):
        '''
        input x: ndarray of shape [N, theta_dim]
        input y: ndarray of shape [N, 1]
        '''
        idx_samples = np.arange(len(x))
        loss = {'train': [], 'validation': []}
        for it in range(iterations):

            # Batches
            np.random.seed(seed)
            np.random.shuffle(idx_samples)
            num_batches = int(np.ceil(len(x) / batch_size))
            loss_epoch = []
            for idx_b in range(num_batches):
                idx_batch = idx_samples[idx_b * batch_size : (idx_b + 1) * batch_size]
                x_batch, y_batch = x[idx_batch], y[idx_batch]

                # Forward and loss
                h = self.forward(x_batch) # shape [batch_size,]
                loss_epoch += [self.loss(h, y_batch, C=C)]
            
                # Compute the gradient
                grad = 1.0 * (y_batch == 0) * (h >= -1) - 1.0 * (y_batch == 1) * (h <= 1)
                grad = C * (np.tile(grad, (1, x_batch.shape[-1])) * x_batch).sum(axis=0, keepdims=True).T
                grad = grad + self.theta

                # Update the gradient
                self.theta = self.theta - lr * grad

            loss['train'] += [np.mean(loss_epoch)]

            # Validation step
            h_val = self.forward(x_val)
            loss['validation'] += [self.loss(h_val, y_val, C=C)]

            # Print log
            if print_every is not None:
                if (it % print_every == 0) or (it == iterations - 1):
                    print(f'Iter: {it}\n\ttrain loss {loss["train"][-1]:.3f}, validation loss {loss["validation"][-1]:.3f}')
        return loss

    def predict(self, x):
        '''
        input x: ndarray of shape [N, theta_dim]
        output y_: ndarray of shape [N, 1]
        '''
        y_ = None
        ##############################
        # INSERT YOUR CODE HERE
        ##############################
        h = self.forward(x)
        y_ = 1 * (h >= 0)
        ##############################
        # END OF YOUR CODE
        ##############################
        return y_

    def evaluate(self, x, y, metric_func):
        '''
        input x: ndarray of shape [N, theta_dim]
        input y: ndarray of shape [N, 1]
        input metric_func: function for evaluation
        '''
        y_ = self.predict(x)
        return metric_func(y_, y)