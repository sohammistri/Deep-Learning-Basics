# Some general utils code, plotting loss values, loss computation etc.

import numpy as np

def mse_loss_and_grads(model, x, y):
    """
        Compute the MSE loss and gradients for a model
        x shape: (B, In)
        y shape: (B, Out=1)
    """
    y_pred = model.forward(x)

    mse_loss = np.sum((y - y_pred)**2) / (2 * x.shape[0])

    ## Steps
    ## loss = np.sum((y-y_pred)**2) / 2B
    ## y_pred = hidden * W  + b # hidden -> (B, H), W -> (H, O), b -> (O)
    
    dy_pred = -(y - y_pred) / (x.shape[0]) # (B, O)
    db = dy_pred.sum(axis=0) # (O)
    dW = hidden * dy_pred

def get_loss_and_grads(model, x, y, loss_fn):
    """
        Get loss and gradients for a MLP model defined previously
    """
    y_pred = model.forward(x)

    loss = loss_fn(y, y_pred)

    dy_pred = 