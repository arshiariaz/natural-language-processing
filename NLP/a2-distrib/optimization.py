import argparse
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def quadratic(x1, x2):
    """
    Quadratic function of two variables
    :param x1: first coordinate
    :param x2: second coordinate
    :return:
    """
    return (x1 - 1) ** 2 + 8 * (x2 - 1) ** 2

def quadratic_grad(x1, x2):
    """
    Returns a numpy array containing the gradient of the quadratic function defined above evaluated at the point
    :param x1: first coordinate
    :param x2: second coordinate
    :return: a one-dimensional numpy array containing two elements representing the gradient
    """
    dx1 = 2 * (x1 - 1)
    dx2 = 16 * (x2 - 1)
    return np.clip(np.array([dx1, dx2]), -1e10, 1e10)

def find_optimal_step_size():
    step_sizes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    best_step_size = None
    best_iterations = float('inf')
    
    for step_size in step_sizes:
        curr_point = np.array([0., 0.])
        iterations = 0
        while np.linalg.norm(curr_point - np.array([1., 1.])) > 0.1 and iterations < 1000:
            grad = quadratic_grad(curr_point[0], curr_point[1])
            curr_point = curr_point - step_size * grad
            iterations += 1
        
        if iterations < best_iterations:
            best_iterations = iterations
            best_step_size = step_size
    
    print(f"Optimal step size: {best_step_size}, Iterations: {best_iterations}")
    return best_step_size

# Call this function to find the optimal step size
OPTIMAL_STEP_SIZE = find_optimal_step_size()

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='optimization.py')
    parser.add_argument('--func', type=str, default='QUAD', help='function to optimize (QUAD or NN)')
    parser.add_argument('--lr', type=float, default=OPTIMAL_STEP_SIZE, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    return args

def sgd_test_quadratic(args):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = quadratic(X, Y)
    plt.figure()

    # Track the points visited here
    points_history = []
    curr_point = np.array([0., 0.])
    for iter in range(0, args.epochs):
        grad = quadratic_grad(curr_point[0], curr_point[1])
        if len(grad) != 2:
            raise Exception("Gradient must be a two-dimensional array (vector containing [df/dx1, df/dx2])")
        next_point = curr_point - args.lr * grad
        points_history.append(curr_point)
        print("Point after epoch %i: %s" % (iter, repr(next_point)))
        curr_point = next_point
    points_history.append(curr_point)
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.plot([p[0] for p in points_history], [p[1] for p in points_history], color='k', linestyle='-', linewidth=1, marker=".")
    plt.title('SGD on quadratic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    exit()

if __name__ == '__main__':
    args = _parse_args()
    sgd_test_quadratic(args)
