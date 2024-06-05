import numpy as np #linear algebra  
import pandas as pd

data = pd.read_csv('mnist_train.csv')

data = np.array(data)

m,n = data.shape() # m for the row, n for the colom

np.random.shuffle(data) #to avoid bias in training

data_test = data[0:1000].T # assigning the first 1000 rows to test set, transpose because coloum is easier to work with than row
y_test = data_test[0] # assigning first colom for labels of training set
X_test = data_test[1:n] # assigning the rest for training set (input parameters)

data_train = data[1000:m].T #assigning rows from 1000 till the end for training set
y_train = data_train[0] # assigning first colom for labels of training set
X_train = data_train[1:n] # assigning the rest for training set (input parameters)

def initial_paramaters():
    # Generate random values for the weights and biases of the layers
    # The shape of the array is (num_neurons, num_inputs)
    # The values are scaled to have a mean of 0 and standard deviation of 1
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    
    # Return the initialized weights and biases
    return W1, b1, W2, b2


def ReLu(Z):
    return np.maximum(0,Z)

def softmax(Z):
    """
    Compute the softmax values for each row of the input array.
    
    Args:
        Z (ndarray): Input array of shape (num_rows, num_cols)
    
    Returns:
        ndarray: Array of shape (num_rows, num_cols) containing the softmax values
    """
    
    # Subtract the maximum value from each row to improve numerical stability
    # This is because the exponential function grows very quickly
    Z -= np.max(Z, axis=1)[:, np.newaxis]
    
    # Compute the exponential of each element in the array
    # This will give us the unnormalized probabilities
    exp_Z = np.exp(Z)
    
    # Compute the sum of the exponential values for each row
    # This will give us the normalizing constant
    sum_exp_Z = np.sum(exp_Z, axis=1)[:, np.newaxis]
    
    # Compute the softmax values by dividing the exponential values by the sum
    # This will give us the probabilities for each class
    A = exp_Z / sum_exp_Z
    
    # Return the softmax values
    return A
    



def forward_prop(W1, b1, W2, b2, X):
    """
    Perform the forward propagation of the neural network.
    
    Args:
        W1 (ndarray): Weights of the first layer, shape (num_neurons, num_inputs)
        b1 (ndarray): Biases of the first layer, shape (num_neurons, 1)
        W2 (ndarray): Weights of the second layer, shape (num_neurons, num_neurons)
        b2 (ndarray): Biases of the second layer, shape (num_neurons, 1)
        X (ndarray): Input data, shape (num_inputs, num_samples)
    
    Returns:
        tuple: Tuple containing the output of each layer:
            - Z1 (ndarray): Input to the first hidden layer, shape (num_neurons, num_samples)
            - A1 (ndarray): Output of the first hidden layer, shape (num_neurons, num_samples)
            - Z2 (ndarray): Input to the output layer, shape (num_neurons, num_samples)
            - A2 (ndarray): Output of the output layer, shape (num_neurons, num_samples)
    """
    
    # Calculate the input to the first hidden layer
    Z1 = W1.dot(X) + b1
    
    # Apply the ReLU activation function to the input to the first hidden layer
    A1 = ReLu(Z1)
    
    # Calculate the input to the output layer
    Z2 = W2.dot(A1) + b2
    
    # Apply the softmax activation function to the input to the output layer
    A2 = softmax(Z2)
    
    # Return the output of each layer
    return Z1, A1, Z2, A2

def ReLu_derivative(Z):
    return Z > 0

def one_hot(Y):
    """
    Convert a numpy array of labels to a one-hot encoded numpy array.
    
    Args:
        Y (ndarray): Input numpy array of labels, shape (num_samples,).
    
    Returns:
        ndarray: One-hot encoded numpy array, shape (num_classes, num_samples).
        
    Raises:
        ValueError: If the input Y is not a numpy array.
    """
    
    # Check if the input Y is a numpy array
    if Y is None or not isinstance(Y, np.ndarray):
        raise ValueError("Input Y must be a numpy array")
    
    # Get the maximum value in the input Y
    num_classes = Y.max() + 1
    
    # Create a numpy array of zeros with shape (num_classes, num_samples)
    one_hot_Y = np.zeros((Y.size, num_classes), dtype=np.int8)
    
    # Set the corresponding rows in the one-hot encoded numpy array to 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    
    # Transpose the one-hot encoded numpy array to shape (num_samples, num_classes)
    one_hot_Y = one_hot_Y.T
    
    # Return the one-hot encoded numpy array
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """
    Perform the backward propagation of the neural network.
    
    Args:
        Z1 (ndarray): Input to the first hidden layer, shape (num_neurons, num_samples)
        A1 (ndarray): Output of the first hidden layer, shape (num_neurons, num_samples)
        Z2 (ndarray): Input to the output layer, shape (num_neurons, num_samples)
        A2 (ndarray): Output of the output layer, shape (num_neurons, num_samples)
        W1 (ndarray): Weights of the first layer, shape (num_neurons, num_inputs)
        W2 (ndarray): Weights of the second layer, shape (num_neurons, num_neurons)
        X (ndarray): Input data, shape (num_inputs, num_samples)
        Y (ndarray): Labels of the data, shape (num_samples,)
    
    Returns:
        tuple: Tuple containing the gradients of the weights and biases:
            - dW1 (ndarray): Gradient of the weights of the first layer, shape (num_neurons, num_inputs)
            - db1 (ndarray): Gradient of the biases of the first layer, shape (num_neurons, 1)
            - dW2 (ndarray): Gradient of the weights of the second layer, shape (num_neurons, num_neurons)
            - db2 (ndarray): Gradient of the biases of the second layer, shape (num_neurons, 1)
    """
    
    # Convert the labels to one-hot encoding
    one_hot_Y = one_hot(Y)
    
    # Calculate the derivative of the cross-entropy loss with respect to the output of the output layer
    dZ2 = A2 - one_hot_Y
    
    # Calculate the gradient of the weights of the second layer
    dW2 = 1 / m * dZ2.dot(A1.T)
    
    # Calculate the gradient of the biases of the second layer
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    # Calculate the derivative of the ReLU activation function
    derivative_ReLU = ReLu_derivative(Z1)
    
    # Calculate the derivative of the cross-entropy loss with respect to the input of the output layer
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU
    
    # Calculate the gradient of the weights of the first layer
    dW1 = 1 / m * dZ1.dot(X.T)
    
    # Calculate the gradient of the biases of the first layer
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    # Return the gradients of the weights and biases
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Update the parameters of the neural network using gradient descent.
    
    Args:
        W1 (ndarray): Weights of the first layer, shape (num_neurons, num_inputs)
        b1 (ndarray): Biases of the first layer, shape (num_neurons, 1)
        W2 (ndarray): Weights of the second layer, shape (num_neurons, num_neurons)
        b2 (ndarray): Biases of the second layer, shape (num_neurons, 1)
        dW1 (ndarray): Gradient of the weights of the first layer, shape (num_neurons, num_inputs)
        db1 (ndarray): Gradient of the biases of the first layer, shape (num_neurons, 1)
        dW2 (ndarray): Gradient of the weights of the second layer, shape (num_neurons, num_neurons)
        db2 (ndarray): Gradient of the biases of the second layer, shape (num_neurons, 1)
        alpha (float): Learning rate
    
    Returns:
        ndarray: Updated weights of the first layer, shape (num_neurons, num_inputs)
        ndarray: Updated biases of the first layer, shape (num_neurons, 1)
        ndarray: Updated weights of the second layer, shape (num_neurons, num_neurons)
        ndarray: Updated biases of the second layer, shape (num_neurons, 1)
    """
    
    # Subtract the learning rate multiplied by the gradients of the weights and biases from the current weights and biases
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2
    
    # Return the updated weights and biases
    return W1, b1, W2, b2


def get_predictions(A2):
    """
    Get the predictions from the output of the neural network.
    
    Args:
        A2 (ndarray): Output of the output layer, shape (num_neurons, num_samples)
    
    Returns:
        ndarray: Predictions, shape (num_samples,)
        
    Explanation:
        - The output of the neural network is a numpy array of shape (num_neurons, num_samples).
        - The `np.argmax` function is used to find the index of the maximum value in each row of the output array.
        - The `axis=0` argument specifies that the maximum value should be found along the rows of the array.
        - The resulting array of indices is the prediction for each sample.
    """
    
    # Get the predictions by finding the index of the maximum value in each row of the output array
    predictions = np.argmax(A2, 0)
    
    # Return the predictions
    return predictions

def get_accuracy(predictions, Y):
    """
    Calculate the accuracy of the predictions made by the neural network.
    
    Args:
        predictions (ndarray): Predictions made by the neural network, shape (num_samples,).
        Y (ndarray): True labels of the data, shape (num_samples,).
    
    Returns:
        float: The accuracy of the predictions, a value between 0 and 1.
        
    Explanation:
        - The `np.sum` function is used to calculate the number of correctly predicted samples.
        - The `==` operator is used to check if the predicted label is equal to the true label.
        - The `np.size` function is used to get the total number of samples.
        - The division of the number of correctly predicted samples by the total number of samples gives the accuracy.
        - The `np.sum` function is used to calculate the number of correctly predicted samples.
        - The `==` operator is used to check if the predicted label is equal to the true label.
        - The `np.size` function is used to get the total number of samples.
        - The division of the number of correctly predicted samples by the total number of samples gives the accuracy.
    """
    
    # Calculate the number of correctly predicted samples
    num_correct = np.sum(predictions == Y)
    
    # Calculate the total number of samples
    total_samples = Y.size
    
    # Calculate the accuracy by dividing the number of correctly predicted samples by the total number of samples
    accuracy = num_correct / total_samples
    
    # Return the accuracy
    return accuracy

def gradient_descent(X, Y, alpha, iterations):
    # Initialize the weights and biases of the neural network
    W1, b1, W2, b2 = initial_paramaters()
    
    # Perform gradient descent for the specified number of iterations
    for i in range(iterations):
        
        # Perform the forward propagation of the neural network
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        
        # Perform the backward propagation of the neural network
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        
        # Update the weights and biases of the neural network using gradient descent
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Print the iteration number and accuracy every 10 iterations
        if i % 10 == 0:
            print("Iteration: ", i)
            
            # Get the predictions made by the neural network
            predictions = get_predictions(A2)
            
            # Calculate the accuracy of the predictions
            accuracy = get_accuracy(predictions, Y)
            
            # Print the accuracy
            print("Accuracy: ", accuracy)
    
    # Return the updated weights and biases of the neural network
    return W1, b1, W2, b2 

W1, b1, W2, b2 = gradient_descent(X_train, y_train, 0.10, 500)

def make_predictions(X, W1, b1, W2, b2):
    """
    Make predictions using the neural network.
    
    Args:
        X (ndarray): Input data, shape (num_inputs, num_samples)
        W1 (ndarray): Weights of the first layer, shape (num_neurons, num_inputs)
        b1 (ndarray): Biases of the first layer, shape (num_neurons, 1)
        W2 (ndarray): Weights of the second layer, shape (num_neurons, num_neurons)
        b2 (ndarray): Biases of the second layer, shape (num_neurons, 1)
    
    Returns:
        ndarray: Predictions made by the neural network, shape (num_samples,)
    """
    
    # Perform the forward propagation of the neural network to get the output of the output layer
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    
    # Get the predictions made by the neural network
    predictions = get_predictions(A2)
    
    # Return the predictions
    return predictions


from matplotlib import pyplot as plt

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

dev_predictions = make_predictions(X_test, W1, b1, W2, b2)
get_accuracy(dev_predictions, y_test)


