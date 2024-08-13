import numpy as np
from tool_function import *
class CustomGradient:
    def __init__(self, mode='mean', quantile=0.5, delta=1.0):
        self.quantile = quantile
        self.delta = delta
        self.mode = mode

    def mean_gradient(self, theta, x):
        
        grad = theta-x
        return grad  

    def quantile_gradient(self, theta, x):
        residual = x - theta
        grad = np.where(residual < 0, (1 - self.quantile), -self.quantile)
        return grad
    
    def quantile_regression_gradient(self, theta, x, y):
        residual = y - np.dot(x, theta)
        grad = np.where(residual < 0, (1 - self.quantile), -self.quantile) * x
        return grad

    def logistic_regression_gradient(self, theta, x, y):
        z = np.dot(theta, x)
        pred = sigmoid(z)
        error = pred - y
        grad = x * error
        return grad

    def huber_loss_gradient(self, theta, x, y):
        residual = y - np.dot(x, theta) 
        is_small_error = np.abs(residual) < self.delta
        small_error_grad = -residual  
        large_error_grad = -self.delta * np.sign(residual)  
        grad = np.where(is_small_error, small_error_grad, large_error_grad) * x
        return grad

    def __call__(self, theta, X, y=None):
        if self.mode == 'mean':
            return self.mean_gradient(theta, X)
        elif self.mode == 'quantile':
            return self.quantile_gradient(theta, X)
        elif self.mode == 'logistic':
            if theta is None:
                raise ValueError("Weights must be provided for logistic regression")
            return self.logistic_regression_gradient(theta, X, y)
        elif self.mode == 'quantile_reg':
            if theta is None:
                raise ValueError("Weights must be provided for quantile regression")
            return self.quantile_regression_gradient(theta, X, y)
        elif self.mode == 'linear_reg':
            if theta is None:
                raise ValueError("Weights must be provided for Huber loss regression")
            return self.huber_loss_gradient(theta, X, y)
        else:
            raise ValueError("Unsupported estimator type: choose 'mean', 'quantile', 'logistic', 'quantile_reg', or 'huber'")