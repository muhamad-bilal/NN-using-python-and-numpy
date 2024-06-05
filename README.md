<h1>Introduction</h1>
This project marks the culmination of my semester-long deep dive into the world of Artificial Intelligence (AI). It's been an exciting journey, starting with the fundamentals and culminating in this hands-on experience of building a neural network capable of recognizing handwritten digits. I've learned so much, and this project is a reflection of my growth and understanding in this fascinating field. 

<h2>Dataset</h2>
The MNIST (Modified National Institute of Standards and Technology) dataset is used for training and evaluating the model. This dataset consists of 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9. Each image is 28x28 pixels in size, and the labels represent the corresponding digit.
Download the dataset from here: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv

<h2>Network Architecture</h2>
The neural network used in this project has the following architecture:

Input Layer: 784 neurons (corresponding to the 28x28 pixel image input)<br>
Hidden Layer: 10 neurons with ReLU activation function<br>
Output Layer: 10 neurons with softmax activation function (for probability distribution over digit classes)<br>
<h2>Code Structure</h2>
init_parameters(): Initializes the network's weights (W1, W2) and biases (b1, b2).<br>
ReLU(Z): Implements the Rectified Linear Unit activation function.<br>
softmax(Z): Implements the softmax activation function.<br>
forward_prop(W1, b1, W2, b2, X): Performs forward propagation through the network.<br>
ReLU_derivative(Z): Computes the derivative of the ReLU function.<br>
one_hot(Y): Converts labels to one-hot encoded representation.<br>
backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y): Performs backpropagation to calculate gradients.<br>
update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): Updates parameters using gradient descent.<br>
get_predictions(A2): Extracts predictions from the network's output.<br>
get_accuracy(predictions, Y): Calculates accuracy.<br>
gradient_descent(X, Y, alpha, iterations): Executes the entire gradient descent training process.<br>
make_predictions(X, W1, b1, W2, b2): Uses the trained model to make predictions.<br>
test_prediction(index, W1, b1, W2, b2): Visualizes a sample prediction.<br>
<h2>Usage</h2>
Make sure you have the MNIST dataset (mnist_train.csv) in the same directory as this code.<br>
Run the script. It will train the network, display training progress and accuracy, and then show a few example predictions.<br>


<h3>Dependencies</h3>
NumPy<br>
Pandas<br>
Matplotlib (for visualization)<br>




