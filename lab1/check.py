import numpy as np

def check(func, print_log=False):

    # Set seed
    np.random.seed(1234)

    # Select function to check
    if func.__name__ == 'predict':
        args = {
            'x': np.random.normal(0, 1, size=(10, 501)),
            'w': np.random.normal(0, 1, size=(501, 1)),
        }
        res = 84.37925870442523
        res_ = func(**args).sum()
        cond = np.absolute(res_.sum() - res) < 1e-8

    elif func.__name__ == 'compute_cost':
        args = {
            'x': np.random.normal(0, 1, size=(10, 501)),
            'y': np.random.normal(0, 1, size=(10, 1)),
            'w': np.random.normal(0, 1, size=(501, 1)),
        }
        res = 131.13466658728424
        res_ = func(**args)
        cond = np.absolute(res_ - res) < 1e-8

    elif func.__name__ == 'compute_cost_multivariate':
        args = {
            'x': np.random.normal(0, 1, size=(10, 501)),
            'y': np.random.normal(0, 1, size=(10, 1)),
            'w': np.random.normal(0, 1, size=(501, 1)),
        }
        res = 131.13466658728424
        res_ = func(**args)
        cond = np.absolute(res_ - res) < 1e-8

    elif func.__name__ == 'gradient_descent':
        args = {
            'x': np.random.normal(0, 1, size=(10, 501)),
            'y': np.random.normal(0, 1, size=(10, 1)),
            'w': np.random.normal(0, 1, size=(501, 1)),
            'learning_rate': 0.005,
            'num_iters': 10,
        }
        res = [
            158.544797156765,
            2.203001889427611,
            25.109538060963843,
        ] # Sums of the arryays
        res_ = func(**args)
        cond = all([np.absolute(r_.sum() - r) < 1e-8 for r, r_ in zip(res, res_)])

    else:
        raise Exception(f'Error. The check of the function {func.__name__} is not implemented.')

    if cond:
        print(f'Your function "{func.__name__}" is correct!')
    else:
        print(f'Your function "{func.__name__}" is NOT correct!')

    # Print output log
    if print_log:
        if isinstance(res, list):
            for r, r_ in zip(res, res_):
                if isinstance(r_, np.ndarray):
                    r_ = r_.sum()
                print(f'Your output: {r_}, expected output: {r}')
        if isinstance(res, float) or isinstance(res, int) or isinstance(res, str):
            print(f'Your output: {res_}, expected output: {res}')