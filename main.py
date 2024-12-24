#!/home/andrew/turic/bin/python3
# Written initially by ChatGPT

import tqdm
import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import glob
from flax.training import train_state
from jax import random


def binary_to_array(num):
    num = str(bin(int(num)))[2:]
    return np.array(list(map(int, num))).reshape((len(num), len(num[0])))

# Load binary arrays from text files
def load_data():
    inputs = []
    targets = []

    i = 0
    for file in tqdm.tqdm(os.listdir("data/"), desc="Processing Data"):
        if i == 10000:
            break
        prime_1, prime_2, semiprime = open("data/" + file).readlines()
        inputs.append(binary_to_array(semiprime))
#        inputs.append(binary_to_array(semiprime))
        targets.append(binary_to_array(prime_1))
#        targets.append(binary_to_array(prime_2))
        i += 1

    inputs = jnp.array(inputs)  # Shape: (num_samples, 32, 16)
    targets = jnp.array(targets)  # Shape: (num_samples, 16, 16)
    return inputs, targets

# Define the CNN model
class ImageModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Reshape input
        x = x.reshape(-1, 32, 16, 1)
        
        # First Conv Layer
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        # Second Conv Layer
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        
        # Flatten
        x = x.reshape(-1, 8 * 4 * 32)
        
        # Dense Layers
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        
        # Output Layer (No sigmoid activation)
        x = nn.Dense(features=16 * 16)(x)
        
        # Reshape output to match target shape
        return x.reshape(-1, 16, 16)

# Create a training state
def create_train_state(rng, learning_rate):
    model = ImageModel()
    variables = model.init(rng, jnp.ones([1, 32 * 16]))
    params = variables['params']  # Extract parameters
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the loss function
@jax.jit
def compute_loss(params, batch):
    variables = {'params': params}
    # Get logits from the model
    logits = ImageModel().apply(variables, batch['X'])  # Shape: (batch_size, 16, 16)
    
    # Ensure that batch['y'] has the same shape as logits
    labels = batch['y'].reshape(-1, 16, 16)  # Shape: (batch_size, 16, 16)
    
    # Compute the loss using logits and labels
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    return loss

# Define the training step
@jax.jit
def train_step(state, batch):
    grads = jax.grad(compute_loss)(state.params, batch)
    return state.apply_gradients(grads=grads)

# Training loop
def train_model(state, X_train, y_train, num_epochs, batch_size):
    num_samples = X_train.shape[0]
    for epoch in range(num_epochs):
        perm = np.random.permutation(num_samples)
        X_train, y_train = X_train[perm], y_train[perm]
        for i in range(0, num_samples, batch_size):
            batch = {
                'X': X_train[i:i+batch_size],
                'y': y_train[i:i+batch_size]
            }
            state = train_step(state, batch)
        loss = compute_loss(state.params, {'X': X_train, 'y': y_train})
        print(f"Epoch {epoch + 1}, Loss: {loss}")
    return state

# Main function
def main():
    # Load data
    X_train, y_train = load_data()

    # Initialize training state
    rng = random.PRNGKey(0)
    state = create_train_state(rng, learning_rate=0.001)

    # Train the model
    state = train_model(state, X_train, y_train, num_epochs=1000, batch_size=32)

    # Save the trained model parameters
    np.save('model_params.npy', state.params)

if __name__ == "__main__":
    main()
