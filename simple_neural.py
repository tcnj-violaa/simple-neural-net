#A 3-layer neural network implementation.
#Much credit to Tariq Rashid's Make Your Own Neural Network book

import numpy
import scipy.special
import csv

class neuralNet:

    def __init__(self, num_inputs, num_hiddens, num_outputs, learning_rate):
        self.inodes = num_inputs
        self.hnodes = num_hiddens
        self.onodes = num_outputs

        #Link weight matrices. Initials sampled from a normal
        #probability distribution
        self.in_to_hidden = numpy.random.normal(0.0, pow(self.hnodes, -0.5),
                (self.hnodes, self.inodes))
        self.hidden_to_out = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                (self.onodes, self.hnodes))

        self.lr = learning_rate

        #Activation function
        self.sigmoid = lambda x: scipy.special.expit(x)

        pass


    def train(self, init_inputs, init_targets):
        inputs = numpy.array(init_inputs, ndmin=2).T
        targets = numpy.array(init_targets, ndmin=2).T

        hidden_inputs = numpy.dot(self.in_to_hidden, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)

        #Signals into the output layer
        final_inputs = numpy.dot(self.hidden_to_out, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.hidden_to_out.T, output_errors)

        #Update weights between hidden and output 
        self.hidden_to_out += self.lr * numpy.dot((output_errors *
            final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs))

        #Between input and hidden
        self.in_to_hidden += self.lr * numpy.dot((hidden_errors *
            hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs))

        pass


    def query(self, init_inputs):
        inputs = numpy.array(init_inputs, ndmin=2).T

        #Signals to/from hidden
        hidden_inputs = numpy.dot(self.in_to_hidden, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)

        #Signals to/from final
        final_inputs = numpy.dot(self.hidden_to_out, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        return final_outputs

    def full_train(self, training_data, epochs):
        data_file = open(training_data, 'r')
        data_list = data_file.readlines()
        data_file.close()

        for e in range(epochs):
            for record in data_list:
                all_values = record.split(',')
                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                targets = numpy.zeros(self.onodes) + 0.01
                targets[int(all_values[0])] = 0.99
                self.train(inputs, targets)
                pass
            pass

    def test(self, testing_data):
        data_file = open(testing_data, 'r')
        data_list = data_file.readlines()
        data_file.close()

        scorecard = []
        correct_values = []
        neural_net_values = []

        for record in data_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            correct_values.append(correct_label)
            #Scale and shift inputs(?)
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = self.query(inputs)
            label = numpy.argmax(outputs)
            neural_net_values.append(label)

            if (label == correct_label):
                scorecard.append(1)

            else:
                scorecard.append(0)
                pass

            pass

        scorecard_array = numpy.asarray(scorecard)

        if correct_values.size < 50:
            print("Correct values:  ", correct_values)
        if neural_net_values.size <= 50:
            print("Our net's values:", neural_net_values)
        if scorecard.size <= 50:
            print("Scorecard:       ", scorecard)

        print("Performance = ", scorecard_array.sum() / scorecard_array.size)


#Simple test without training; currently meaningless
#num_input_nodes = 3
#num_hidden_nodes = 50
#num_output_nodes = 3

#learning_rate = 0.4

test_net = neuralNet(784, 200, 10, 0.1)

test_net.full_train("mnist-data/mnist_train_full.csv", 5)
test_net.test("mnist-data/mnist_test_full.csv")
test_net.test("mnist-data/mnist_test_10.csv")


