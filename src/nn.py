"""
CLASS MLP written by Van Nipper
Constructor: Number of inputs in first layer of Neural Network
A class which allows the user to create a Mult-Layer Perceptron for use
"""

import numpy as np
from scipy.special import expit

class MLP:

    def __init__(self, input_size):
        """
        MLP CONSTRUCTOR
        Input: An integer which represents the number of nodes in the input layer
        Output: Neural network object
        Creates the first layer of the neural network, as well as initializes the
        lists for weights, biases, activations, outputs, errors, and deltas.
        """
        
        # Initialize an integer list for the number of nodes in each layer
        self.layers_sizes = [input_size]

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # Initialize activations and outputs
        self.activations = []
        self.outputs = []

        # Initialize errors and deltas
        self.errors = []
        self.deltas = []

    def addLayer(self, size):
        """
        FUNCTION addLayer
        Input: The desired size of the new layer
        Output: None (void)
        Adds an additional layer to the neural network. The new layer becomes the output
        layer until this function is called again.
        """
        
        # Add the number of nodes to the list of layer sizes
        self.layers_sizes.append(size)

        # Add weights and biases using np
        self.weights.append(np.random.randn(self.layers_sizes[-2], size) * np.sqrt(1 / self.layers_sizes[-2])) # Xavier for sigmoid
        
        if len(self.layers_sizes) > 2:

            self.weights[-2] = (np.random.randn(self.layers_sizes[-3], self.layers_sizes[-2]) * np.sqrt(2 / self.layers_sizes[-3])) # Set previous layer to He for relu

        self.biases.append(np.zeros((1, size)))

        # Re-initialize activations and outputs
        self.activations.append(np.zeros((1, size)))
        self.outputs.append(np.zeros((1, size)))

        # Re-initialize errors and deltas
        self.errors.append(np.zeros((1, size)))
        self.deltas.append(np.zeros((1, size)))

    def feedForward(self, X):
        """
        FUNCTION feedFoward
        Input: X, or the input vector for whatever data that's being observed
        Output: The vector from the output layer after all values pass through the
        network
        Feeds an input vector through the neural network. It does this by multiplying each
        activation from the previous layer with all weights in a given row, then adding a
        bias. It then performs the sigmoid activation function to condense the new activations
        into the range 0 - 1.
        """

        # Define sigmoid activation function
        sigmoid = expit
        relu = lambda x: np.maximum(0, x)

        # Iterate through all the layers in the network
        for layerNum in range(0, len(self.layers_sizes) - 1):

            # If on the first layer, use input vector X
            if layerNum == 0:
                self.activations[0] = np.dot(X, self.weights[0]) + self.biases[0] # wx + b
                self.outputs[0] = relu(self.activations[0]) # relu

            # if on the last layer, use sigmoid
            elif layerNum == len(self.layers_sizes) - 2:
                self.activations[layerNum] = np.dot(self.outputs[layerNum - 1], self.weights[layerNum]) + self.biases[layerNum] # wx + b
                self.outputs[layerNum] = sigmoid(self.activations[layerNum]) # sigmoid
            
            # Otherwise, use the previous layer's output vector and relu
            else:
                self.activations[layerNum] = np.dot(self.outputs[layerNum - 1], self.weights[layerNum]) + self.biases[layerNum] # wx + b
                self.outputs[layerNum] = relu(self.activations[layerNum]) # relu

        # Return the vector of outputs from the last layer
        return self.outputs[-1]

    def train(self, Xset, Yset, epochs, learning_rate, Xtest, Ytest):
        """
        FUNCTION train
        Inputs: Xset (dataset), Yset (labels set), epochs, learning_rate (change of weights/biases 
        per epoch)
        Output: None (void)
        Trains the neural network by feeding an input vector through, then examining the difference
        between the expected and actual outcome to adjust weights and biases in each layer. Propagates 
        backwards to update weights and biases. Prints training results occasionally as NN is being trained.
        """
        
        # Sigmoid derivative function
        sigmoid_derivative = lambda x: np.clip(x * (1 - x), 1e-8, 1e8)
        relu_derivative = lambda x: (x > 0).astype(float)

        for epoch in range(epochs):

            photoCount = 0
            successCount = 0
            batchcount = 0

            # Iterate through each batch
            for batch, Ybatch in zip(Xset, Yset):

                batchcount += 1
                
                # Reset weight and bias updates for the batch
                weight_updates = [np.zeros_like(w) for w in self.weights]
                bias_updates = [np.zeros_like(b) for b in self.biases]
                loss = 0  # Reset loss per batch
                
                # Process each sample in the batch
                for data, label in zip(batch, Ybatch):
                    
                    # Forward Pass
                    data = data.reshape(1, -1)
                    output = self.feedForward(data)
                    
                    # Expected output
                    expected_output = np.zeros((1, self.layers_sizes[-1]))
                    expected_output[0][int(label - 1)] = 1  

                    # Find error and loss
                    self.errors[-1] = expected_output - output
                    loss += np.mean((output - expected_output) ** 2)

                    # Calculating epoch accuracy
                    max_output = 0
                    for i in range(len(output[0])):
                        if output[0][i] > output[0][max_output]:
                            max_output = i
                    
                    if max_output == (int(label - 1)):
                        successCount += 1
                    photoCount += 1

                    # Back propagation
                    for layerNum in range(len(self.layers_sizes) - 2, -1, -1): # BUG

                        # Weight and Bias Update Accumulation
                        if layerNum == 0:

                            self.deltas[layerNum] = self.errors[layerNum] * relu_derivative(self.outputs[layerNum])
                            weight_updates[layerNum] += np.dot(data.T, self.deltas[layerNum])

                        # sigmoid if last layer
                        elif layerNum == len(self.layers_sizes) - 2:

                            self.deltas[layerNum] = self.errors[layerNum] * sigmoid_derivative(self.outputs[layerNum])
                            weight_updates[layerNum] += np.dot(self.outputs[layerNum - 1].T, self.deltas[layerNum])

                        # relu otherwise
                        else:

                            self.deltas[layerNum] = self.errors[layerNum] * relu_derivative(self.outputs[layerNum])
                            weight_updates[layerNum] += np.dot(self.outputs[layerNum - 1].T, self.deltas[layerNum])

                        bias_updates[layerNum] += np.sum(self.deltas[layerNum], axis=0, keepdims=True)

                        # Compute errors for next layer (except input layer)
                        if layerNum > 0:
                            self.errors[layerNum - 1] = np.dot(self.deltas[layerNum], self.weights[layerNum].T)

                # Apply weight and bias updates for the batch
                for layerNum in range(len(self.layers_sizes) - 1):

                    self.weights[layerNum] += (weight_updates[layerNum] / len(batch)) * learning_rate
                    self.biases[layerNum] += (bias_updates[layerNum] / len(batch)) * learning_rate

            # Test the model on the test set
            test_successCount = 0
            test_photoCount = 0
            test_loss = 0
            for test_data, test_label in zip(Xtest, Ytest):
                    # Forward Pass
                    test_data = test_data.reshape(1, -1)
                    output = self.feedForward(test_data)
                    
                    # Expected output
                    expected_output = np.zeros((1, self.layers_sizes[-1]))
                    expected_output[0][int(test_label - 1)] = 1  
                    test_loss += np.mean((output - expected_output) ** 2)
                    # Calculating epoch accuracy
                    max_output = 0
                    for i in range(len(output[0])):
                        if output[0][i] > output[0][max_output]:
                            max_output = i
                    
                    if max_output == (int(test_label - 1)):
                        test_successCount += 1
                    test_photoCount += 1

            # Print loss results for the epoch
            print(f"Epoch {epoch + 1}, Accuracy: {(float(successCount/photoCount))*100:.2f}% Loss: {loss / len(Xset):.4f}; Test Accuracy: {(float(test_successCount/test_photoCount))*100:.2f}% Loss: {test_loss / len(Xtest):.4f}")

    def save (self, filename):
        """
        FUNCTION save
        Input: filename (string)
        Output: None (void)
        Saves the weights and biases of the neural network to a file
        """
    
        file = open(filename, 'w')
        
        # Save number of nodes in each layer to file as a string
        for val in self.layers_sizes:

            file.write(f"{val} ")
        
        file.write('\n')

        # Save weights to file
        for i in range(len(self.weights)):

            for j in range(len(self.weights[i])):

                for k in range(len(self.weights[i][j])):
                    
                    # Write the weights to the file
                    file.write(f"{self.weights[i][j][k]} ")

                # Add a line break between each row
                file.write('\n')

        # Save biases to file
        for i in range(len(self.biases)):
            
            for j in range(len(self.biases[i][0])):

                # Write the biases to the file
                file.write(f"{self.biases[i][0][j]} ")

            # Add a line break between each vector
            file.write('\n')
        
        file.close()

    def load (self, filename):
        """
        FUNCTION load
        Input: filename (string)
        Output: None (void)
        Loads the weights and biases of the neural network from a file
        dataType -> List of numpy array of numpy array (weights)
        dataType -> List of numpy array (biases)
        """

        # Read number of nodes in each layer into temporary list
        file = open(filename, 'r')
        tempLayers = list(map(int, file.readline().split()))

        # Return if the number of nodes in the input layer is different or if another
        # layer is initialized
        if tempLayers[0] != self.layers_sizes[0] or len(self.layers_sizes) != 1:
            print(f'Could not load from file {filename} for this model.')
            return -1

        """ READ IN WEIGHTS """
        # For each layer
        for i in range(len(tempLayers) - 1):
            
            # Initialize numpy array of zeros with correct dimensions
            self.weights.append(np.zeros((tempLayers[i], tempLayers[i + 1])))

            # For each row in the weights matrix
            for j in range(tempLayers[i]):

                # Read the row data in from the file
                templine = list(map(float, file.readline().split()))

                # For each column in that row
                for k in range(tempLayers[i + 1]):

                    # Set the weight to the value in the file
                    self.weights[i][j][k] = templine[k]

        """ READ IN BIASES """
        # For each layer
        for i in range(len(tempLayers) - 1):
            
            # Initialize numpy array of zeros with correct dimensions
            self.biases.append(np.zeros((1, tempLayers[i + 1])))

            # Read the biases vector in from the file
            templine = list(map(float, file.readline().split()))

            # For each value in the biases vector
            for j in range(tempLayers[i + 1]):

                # Set the bias to the value in the file
                self.biases[i][0][j] = templine[j]

            # Re-initialize activations and outputs
            self.activations.append(np.zeros((1, tempLayers[i + 1])))
            self.outputs.append(np.zeros((1, tempLayers[i + 1])))

            # Re-initialize errors and deltas
            self.errors.append(np.zeros((1, tempLayers[i + 1])))
            self.deltas.append(np.zeros((1, tempLayers[i + 1])))

            # Adjust the number of nodes in each layer
            self.layers_sizes = tempLayers

        file.close()
        return 0