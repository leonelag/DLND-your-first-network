import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))

        self.lr = learning_rate
        
        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1.0 / (1.0 + np.exp(-x))
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        # Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        # in output layer, use identity function as activation
        final_outputs = final_inputs

        # Output error
        output_errors = targets - final_outputs
        
        # TODO: Backpropagated error
        # errors propagated to the hidden layer
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
        # hidden layer gradients
        hidden_grad   = hidden_outputs * (1 - hidden_outputs)

        # TODO: Update the weights
        _, n_records = inputs.shape
        self.weights_hidden_to_output += self.lr \
                                         * np.dot(output_errors, hidden_outputs.T) \
                                         / n_records
        
        # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden  += self.lr \
                                        * np.dot(hidden_errors * hidden_grad, inputs.T) \
                                        / n_records

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        # Hidden layer
        hidden_inputs  = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = np.array([self.activation_function(input) 
                                   for input in hidden_inputs],
                                  ndmin=2)

        # signals into final output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        # in output layer, use identity function as activation
        final_outputs = final_inputs

        return final_outputs
 
