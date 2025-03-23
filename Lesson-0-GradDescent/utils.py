# Some general utils code, plotting loss values, loss computation etc.

import numpy as np
import matplotlib.pyplot as plt 

def mse_loss(y, y_pred):
    """
        Compute MSE loss for given pair of y and y_pred
        Expected numpy array for both
    """
    B = y.shape[0]
    loss = np.sum((y - y_pred) ** 2) / (2 * B)
    dy_pred = -(y - y_pred) / B
    return loss, dy_pred

def nll_loss(y, y_pred):
    """
        Compute NLL loss for given pair of y and y_pred
        Expected numpy array for both
        y shape: (B, 1) -> list of classification labels
        y_pred shape: (B, O) -> o/p logits
    """
    B = y.shape[0]
    # Scale logits
    max_logits = y_pred.max(axis=1, keepdims=True)
    logits_scaled = y_pred - max_logits
    # Exp and get the counts
    counts = np.exp(logits_scaled)
    sum_counts = counts.sum(axis=1, keepdims=True)
    # get the probs
    probs = counts / sum_counts
    # target probs
    target_probs = probs[np.arange(B), y.flatten()]
    # get the loss
    loss = -np.log(target_probs).sum() / B

    # compute the gradient
    dy_pred = probs.copy()
    dy_pred[np.arange(B), y.flatten()] -= 1.0

    return loss, dy_pred

def get_accuracy(y, y_pred):
    B = y.shape[0]
    pred_classes = y_pred.argmax(axis=1, keepdims=True)
    acc = (pred_classes == y).sum() / B
    return acc

def plot_loss(train_loss_values, val_loss_values):
    plt.plot(range(len(train_loss_values)), train_loss_values, label="Train Loss") 
    plt.plot(range(len(val_loss_values)), val_loss_values, label="Val Loss") 

    # plt.scatter(range(len(loss_values)), loss_values, c='red', s=20)

    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_one_epoch(model, xb, yb, optim, loss_fn, acc=False):
    """
        Function to see what happens in a single epoch of training
        We take the model, xb and yb are the mini batches, the optim object and the loss_fn
    """
    model.reset_grads_to_zero()
    yb_pred = model.forward(xb) # forward
    loss, dyb_pred = loss_fn(yb, yb_pred) # compute loss and get the dy_pred
    model.backward(xb, dyb_pred) # compute the gradients
    params, grads = model.get_params(), model.get_grads() # flatten the params and grads
    updated_params = optim.update(params, grads) # update as per our optim
    model.update_params(updated_params) # update params

    accuracy = None
    if acc:
        accuracy = get_accuracy(yb, yb_pred)
    return loss, accuracy

def create_mini_batches(x_train, y_train, batch_size):
    """
    Create mini-batches from training data

    Parameters:
    -----------
    x_train : numpy array of shape (n_samples, input_dim)
        Training features
    y_train : numpy array of shape (n_samples, output_dim)
        Training labels
    batch_size : int
        Size of each mini-batch

    Returns:
    --------
    list of tuples
        Each tuple contains (x_batch, y_batch) for one mini-batch
    """
    # Get the number of training examples
    n_samples = x_train.shape[0]

    # Shuffle the training data
    indices = np.random.permutation(n_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]

    # Create mini-batches
    mini_batches = []

    # Complete mini-batches
    num_complete_batches = n_samples // batch_size
    for i in range(num_complete_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        x_batch = x_shuffled[start_idx:end_idx].copy()  # Using copy to avoid memory sharing
        y_batch = y_shuffled[start_idx:end_idx].copy()

        mini_batches.append((x_batch, y_batch))

    # Handle the remaining examples (if any)
    if n_samples % batch_size != 0:
        start_idx = num_complete_batches * batch_size

        x_batch = x_shuffled[start_idx:].copy()
        y_batch = y_shuffled[start_idx:].copy()

        mini_batches.append((x_batch, y_batch))

    return mini_batches

def train(model, x_train, y_train, x_val, y_val, optim, loss_fn, n_epochs, batch_size):
    """
        Unified function for training
    """
    n_samples = x_train.shape[0]
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        # Create mini-batches
        mini_batches = create_mini_batches(x_train, y_train, batch_size)

        epoch_loss = 0

        # Train on mini-batches
        for x_batch, y_batch in mini_batches:
            batch_loss, _ = train_one_epoch(model, x_batch, y_batch, optim, loss_fn)
            epoch_loss += batch_loss * len(x_batch) / n_samples

        train_losses.append(epoch_loss)

        # Compute val loss
        y_val_pred = model.forward(x_val)
        val_loss, _ = loss_fn(y_val, y_val_pred)
        val_losses.append(val_loss)

        # Print progress
        if ((epoch + 1) % 10 == 0) or (epoch == n_epochs - 1):
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses

def train_with_acc(model, x_train, y_train, x_val, y_val, optim, loss_fn, n_epochs, batch_size):
    """
        Unified function for training
    """
    n_samples = x_train.shape[0]
    train_losses, train_acc, val_losses, val_acc = [], [], [], []

    for epoch in range(n_epochs):
        # Create mini-batches
        mini_batches = create_mini_batches(x_train, y_train, batch_size)

        epoch_loss = 0
        total_acc = 0

        # Train on mini-batches
        for x_batch, y_batch in mini_batches:
            batch_loss, accuracy = train_one_epoch(model, x_batch, y_batch, optim, loss_fn, acc=True)
            epoch_loss += batch_loss * len(x_batch) / n_samples
            total_acc += accuracy * len(x_batch) / n_samples

        train_losses.append(epoch_loss)
        train_acc.append(total_acc)

        # Compute val loss
        y_val_pred = model.forward(x_val)
        val_loss, _ = loss_fn(y_val, y_val_pred)
        val_accuracy = get_accuracy(y_val, y_val_pred)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

        # Print progress
        if (epoch == 0) or ((epoch + 1) % 50 == 0) or (epoch == n_epochs - 1):
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {epoch_loss:.6f}, Train Accuracy: {total_acc * 100.00:.2f} %, Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy * 100.00:.2f} %")

    return train_losses, train_acc, val_losses, val_acc