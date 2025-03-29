import numpy as np

# Load the original MNIST dataset
with np.load('mnist.npz', allow_pickle=True) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

print(f"Original shapes: x_train {x_train.shape}, y_train {y_train.shape}, x_test {x_test.shape}, y_test {y_test.shape}")

# Reshape and normalize as needed
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255

print(f"Reshaped: x_train {x_train.shape}, x_test {x_test.shape}")

# Save with the expected key names
np.savez('mnist.npz', 
         train_images=x_train,
         train_labels=y_train,
         test_images=x_test,
         test_labels=y_test)

print("MNIST dataset saved with correct keys") 