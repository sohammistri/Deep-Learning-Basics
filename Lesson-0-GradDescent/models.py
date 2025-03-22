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
        self.V = np.random.randn((self.input_dim, self.hidden_dim))
        self.a = np.random.randn(self.hidden_dim)
        self.W = np.random.randn((self.hidden_dim, self.output_dim))
        self.b = np.random.randn(self.output_dim)

    def forward(self, x):
        """
            Forward pass of MLP
            Expect x to be of shape (B, input_dim). Specially last dim should be input_dim
        """
        assert x.shape[-1] == self.input_dim

        hidden = x @ self.V + self.a # (B, in) @ (in, hid) -> (B, hid)
        activations = np.tanh(hidden) # (B, hid)
        out = activations @ self.W + self.b # (B, hid) @ (hid, out) -> (B, out)

        return out
    
    def get_params(self):
        # Flatten all parameters and concatenate them
        return np.concatenate([
            self.V.flatten(),  # Flatten the input-to-hidden weights
            self.a.flatten(),  # Hidden layer bias
            self.W.flatten(),  # Hidden-to-output weights
            self.b.flatten()   # Output layer bias
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

