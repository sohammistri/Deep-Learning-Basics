# This file contains all GD optimizers. The goal is to take a np array of params and grads and implement just the update

import numpy as np

class GDOptimizer:
    def __init__(self):
        pass

    def update(self, params, grad):
        """
        Update the parameters using the provided gradients.
        Args:
            params (np.ndarray): 1D array of all parameters.
            grad (np.ndarray): 1D array of gradients of the parameters, same shape as `params`.
        Returns:
            params (np.ndarray): updated params.
        """
        pass

class SGDOptimizer(GDOptimizer):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def update(self, params, grad):
        params -= self.lr * grad
        return params
    
class SGDWithMomentum(GDOptimizer):
    def __init__(self, lr, gamma):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.v = None

    def update(self, params, grad):
        if self.v is None:
            self.v = np.zeros(params.shape, dtype=np.float64)
        self.v = self.gamma * self.v + self.lr * grad
        params -= self.v
        return params
    
class NAG(GDOptimizer):
    def __init__(self, lr, gamma, grad_fn):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.grad_fn = grad_fn
        self.v = None

    def update(self, params, grad):
        if self.v is None:
            self.v = np.zeros(params.shape, dtype=np.float64)
        self.v = self.gamma * self.v + self.lr * self.grad_fn(params - self.gamma * self.v)
        params -= self.v
        return params
    
class AdaGrad(GDOptimizer):
    def __init__(self, lr, epsilon):
        super().__init__()
        self.lr = lr
        self.epsilon = epsilon
        self.alpha = None
        self.lr_list = []

    def update(self, params, grad):
        if self.alpha is None:
            self.alpha = np.zeros(params.shape, dtype=np.float64)
        self.alpha += grad**2
        lr_scaled = self.lr / np.sqrt(self.alpha + self.epsilon)
        self.lr_list.append(lr_scaled)
        params -= lr_scaled * grad
        return params
    
class RMSProp(GDOptimizer):
    def __init__(self, lr, epsilon, beta):
        super().__init__()
        self.lr = lr
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = None
        self.lr_list = []

    def update(self, params, grad):
        if self.alpha is None:
            self.alpha = np.zeros(params.shape, dtype=np.float64)
        self.alpha = self.beta * self.alpha + (1 - self.beta) * grad**2
        lr_scaled = self.lr / np.sqrt(self.alpha + self.epsilon)
        self.lr_list.append(lr_scaled)
        params -= lr_scaled * grad
        return params
    
class AdaDelta(GDOptimizer):
    def __init__(self, epsilon, beta):
        super().__init__()
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = None
        self.delta_x = None
        self.lr_list = []

    def update(self, params, grad):
        if self.alpha is None:
            self.alpha = np.zeros(params.shape, dtype=np.float64)
        if self.delta_x is None:
            self.delta_x = np.zeros(params.shape, dtype=np.float64)
        self.alpha = self.beta * self.alpha + (1 - self.beta) * grad**2
        lr_scaled = np.sqrt((self.delta_x + self.epsilon) / (self.alpha + self.epsilon))
        self.lr_list.append(lr_scaled)
        delta_theta = lr_scaled * grad
        params -= delta_theta
        self.delta_x = self.beta * self.delta_x + (1 - self.beta) * delta_theta**2
        return params
        
class Adam(GDOptimizer):
    def __init__(self, lr, epsilon, beta1, beta2):
        super().__init__()
        self.lr = lr
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 1
        self.lr_list = []

    def update(self, params, grad):
        if self.m is None:
            self.m = np.zeros(params.shape, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros(params.shape, dtype=np.float64)

        # update m and v
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        # correct bias
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Get scaled lr
        lr_scaled = self.lr / (np.sqrt(v_hat) + self.epsilon)
        self.lr_list.append(lr_scaled)

        # Update params
        params -= lr_scaled * m_hat

        # Update timestep
        self.t += 1
        return params
    
class AdaMax(GDOptimizer):
    def __init__(self, lr, epsilon, beta1, beta2):
        super().__init__()
        self.lr = lr
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 1
        self.lr_list = []

    def update(self, params, grad):
        if self.m is None:
            self.m = np.zeros(params.shape, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros(params.shape, dtype=np.float64)

        # update m and v
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = np.maximum(self.beta2 * self.v, np.abs(grad))

        # correct bias
        m_hat = self.m / (1 - self.beta1**self.t)

        # Get scaled lr
        lr_scaled = self.lr / (self.v + self.epsilon)
        self.lr_list.append(lr_scaled)

        # Update params
        params -= lr_scaled * m_hat

        # Update timestep
        self.t += 1
        return params

class NAdam(GDOptimizer):
    def __init__(self, lr, epsilon, beta1, beta2):
        super().__init__()
        self.lr = lr
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 1
        self.lr_list = []

    def update(self, params, grad):
        if self.m is None:
            self.m = np.zeros(params.shape, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros(params.shape, dtype=np.float64)

        # update m and v
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        # correct bias
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Get scaled lr
        lr_scaled = self.lr / (np.sqrt(v_hat) + self.epsilon)
        self.lr_list.append(lr_scaled)

        # Update params
        params -= lr_scaled * (self.beta1 * m_hat + ((1 - self.beta1) / (1 - self.beta1**self.t)) * grad)

        # Update timestep
        self.t += 1
        return params