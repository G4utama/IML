import numpy as np
#import matplotlib.pyplot as plt
#plt.style.use('seaborn')

# Check function. Used for checking your code, you can ignore this.
def check_func(func, *args):
    res = {
        'sigmoid': np.array([4.5397868702434395e-05, 0.0066928509242848554, 
                   0.11920292202211755, 0.5, 0.8807970779778823, 
                   0.9933071490757153, 0.9999546021312976]).reshape(-1, 1),
        'xent': [1.3758919771597742, 0.8487364948617685, 0.8616020843171404, 
                    1.2847647859647024, 1.0979517701821886, 1.2448204497955682, 
                    1.148747135298692, 0.9142123727250151, 0.9503784164146648, 
                    0.5259295090148516],
        'gradient': [0.05517542101762578, 0.056956027283867096, 
                     0.10246057091641199, 0.1729957759994022, 
                     -0.07420252599770667, -0.10440419638201064, 
                     0.0814283804697549, -0.1722842404987195, 
                     0.16445779692602908, -0.0962341984706773],
        'predict_lc': [1., -1., 1., 1., -1., -1., 1., 1., -1., 1.],
        'predict_lr': [1., 0., 1., 1., 0., 0., 1., 1., 0., 1.],
        'mse': [0.20665046013997404, 0.12503917069875575, 0.15871653661209398,
                0.24302395420833073, 0.19655530689347242, 0.19455858448384966,
                0.2078562677711177, 0.1626431893970271, 0.12018726319497597, 
                0.0969268504978511]
    }
    with open('./ex2data1.txt', 'r') as txt:
        y = np.array([[float(line.strip().split(',')[2])] 
                      for line in txt.readlines()], dtype=np.float32)
    res['evaluate_lr'] = [y]
    res['evaluate_lc'] = [np.array([-1 if y_i < 0.5 else 1 for y_i in y.reshape(-1)], dtype=np.float32).reshape(-1, 1)]
    print('CHECK RESULTS:')
    print('\n' + ''.join(['=' for _ in range(40)]))
    are_correct = []
    for idx, y in enumerate(res[func.__name__]):
        arg = [a[idx] for a in args]
        y_ = func(*arg)
        if func.__name__ == 'evaluate_lr':
            acc = (1.0 * (y_ == y)).mean()
            y_, y = acc, 0.88
            are_correct += [f'{y_:.4f}' > f'{y:.4f}']
            print(f'Your train accuracy: {100 * y_:.2f}%, Expected train accuracy: > {100 * y:.2f}%')
        elif func.__name__ == 'evaluate_lc':
            acc = (1.0 * (y_ == y)).mean()
            y_, y = acc, 0.88
            are_correct += [f'{y_:.4f}' > f'{y:.4f}']
            print(f'Your train accuracy: {100 * y_:.2f}%, Expected train accuracy: > {100 * y:.2f}%')
        else:
            if isinstance(y_, np.ndarray):
                y_ = y_.reshape(-1)[0]
            if isinstance(y, np.ndarray):
                y = y.reshape(-1)[0]
            are_correct += [f'{y_:.4f}' == f'{y:.4f}']
            print(f'Your result: {y_:.4f}, Expected: {y:.4f}')
    print('\n' + ''.join(['=' for _ in range(40)]))
    if all(are_correct):
        print('Function is correct! Well done!'.upper())
    else:
        print('Function is not correct. Find the bug.')