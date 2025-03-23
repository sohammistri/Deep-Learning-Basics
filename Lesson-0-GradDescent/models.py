# Implement a MLP from scratch using numpy
# We restrict to only one hidden layer though

import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        np.random.seed(42)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # format: out = W*ReLU(V*x+a)+b
        self.V = np.random.randn(self.input_dim, self.hidden_dim)
        self.a = np.random.randn(self.hidden_dim)
        self.W = np.random.randn(self.hidden_dim, self.output_dim)
        self.b = np.random.randn(self.output_dim)

        self.dV = np.zeros_like(self.V, dtype=np.float64)
        self.da = np.zeros_like(self.a, dtype=np.float64)
        self.dW = np.zeros_like(self.W, dtype=np.float64)
        self.db = np.zeros_like(self.b, dtype=np.float64)

        self.hidden = None
        self.layer_1_out = None
        self.out = None

    def forward(self, x):
        """
            Forward pass of MLP
            Expect x to be of shape (B, input_dim). Specially last dim should be input_dim
        """
        assert x.shape[-1] == self.input_dim
        self.hidden = x @ self.V + self.a # (B, in) @ (in, hid) -> (B, hid)
        self.layer_1_out = np.tanh(self.hidden) # (B, hid)
        self.out = self.layer_1_out @ self.W + self.b # (B, hid) @ (hid, out) -> (B, out)

        return self.out
    
    def backward(self, x, dy_hat):
        """
            Compute the grad variables for the params
            We take the dy_hat which will be of shape (B, O)
            This will be calculated based on loss
        """
        # y_hat = W * layer_1_out + b
        self.db = dy_hat.sum(axis=0).reshape(self.b.shape) # dy_hat (B,O) -> db (O)
        self.dW = self.layer_1_out.T @ dy_hat # layer_1_out.T -> (H, B), dy_hat (B, O) W shape (H, O)
        dlayer_1_out = dy_hat @ self.W.T # dy_hat (B, O) W.T shape (O, H)

        # self.layer_1_out = np.tanh(self.hidden)
        dhidden = (1 - self.layer_1_out**2) * dlayer_1_out # hidden shape -> (B, H)

        # self.hidden = x @ self.V + self.a
        self.da = dhidden.sum(axis=0).reshape(self.a.shape) # dhidden (B, H); a shape (H)
        self.dV = x.T @ dhidden # dhidden (B, H) , x shape (B, I) V shape (I, H)
    
    def get_params(self):
        return np.concatenate([
            self.V.flatten(),  # Flatten the input-to-hidden weights
            self.a.flatten(),  # Hidden layer bias
            self.W.flatten(),  # Hidden-to-output weights
            self.b.flatten()   # Output layer bias
        ])
    
    def get_params_count(self):
        return self.V.size + self.a.size + self.W.size + self.b.size
    
    def get_grads(self):
        return np.concatenate([
            self.dV.flatten(),  # Flatten the input-to-hidden weights
            self.da.flatten(),  # Hidden layer bias
            self.dW.flatten(),  # Hidden-to-output weights
            self.db.flatten()   # Output layer bias
        ])
    
    def update_params(self, flat_params):
        # Calculate the sizes of each parameter array/matrix
        v_size = self.input_dim * self.hidden_dim
        a_size = self.hidden_dim
        w_size = self.hidden_dim * self.output_dim

        # Split and reshape parameters
        start = 0

        # Update V (input to hidden weights)
        self.V = flat_params[start:start + v_size].reshape(self.input_dim, self.hidden_dim)
        start += v_size

        # Update a (hidden bias)
        self.a = flat_params[start:start + a_size]
        start += a_size

        # Update W (hidden to output weights)
        self.W = flat_params[start:start + w_size].reshape(self.hidden_dim, self.output_dim)
        start += w_size

        # Update b (output bias)
        self.b = flat_params[start:]
        
    def reset_grads_to_zero(self):
        self.dV = np.zeros_like(self.V, dtype=np.float64)
        self.da = np.zeros_like(self.a, dtype=np.float64)
        self.dW = np.zeros_like(self.W, dtype=np.float64)
        self.db = np.zeros_like(self.b, dtype=np.float64)

