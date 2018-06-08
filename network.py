"""

This module implements the momentum based stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.

"""

import random
import sys
import math
import csv
import cPickle
import time
from collections import Counter

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool

# Activation functions for neurons
from theano.tensor.nnet import sigmoid

def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)


# theano trying to use the gpu
try: theano.config.device = 'gpu'
except: pass # it's already set
theano.config.floatX = 'float32'



class Trainer(object):
    """

    A class to automatically optimize hyperparameters

    """

    def __init__(self, training_data, validation_data, test_data, eta, lmbda, my, max_epochs, writer, mini_batch_size, dropout, net_name):
        """

        initialize with network hyperparameters

        """
        self.eta = eta
        self.lmbda = lmbda
        self.my = my
        self.max_epochs = max_epochs
        self.writer = writer
        self.mini_batch_size = mini_batch_size
        self.net = None
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.dropout = dropout
        self.net_name = net_name


    def train(self, early_stopping=True):
        """

        instantiate a new network if necessary and perform momentum based stochastic gradient descent to train it

        """

        print " "
        print " "
        print "start training runs with eta: {0}, lambda: {1}, momentum: {2}, dropout: {3}".format(self.eta, self.lmbda, self.my, self.dropout)

        if self.net is None:
            self.net = Network(getLayers(self.mini_batch_size, self.dropout), self.writer, self.mini_batch_size, self.net_name)

        self.net.MBSGD(self.training_data, self.validation_data, self.test_data, self.max_epochs, self.eta, self.lmbda, self.my, early_stopping)

        return


    def optimizeParameter(self, runs, attribute):
        """

        performs several training runs on the network with different hyperparameters to
        evaluate their effect on accuracy

        the algorithm tests 3 values, distributed over a reasonable range

        after every run, 2 new values are introduced based on the performance of existing ones to find a local optimum

        """


        # setting up basic properties

        results = [0, 0, 0]                                     # init results

        current_value = self.get_attribute(attribute)           # get the networks current value for the given parameter
        if current_value == 0:
            current_value = 1

        scales = self.get_scales(attribute)                     # get predefined scales to create a reasonable range of
                                                                #   values to test

        values = [current_value * scales[0], current_value * scales[1], current_value * scales[2]]



        # check if current setup has already been tested to avoid unnecessary training
        # (usually this applies only when optimizing multiple parameters in succession)

        if self.net != None and len(self.net.trainingResults.validationAccuracy) > 0:

            for i in range(len(values)):

                if values[i] == current_value:
                    results[i] = max(self.net.trainingResults.validationAccuracy)
                    break


        # optimizing

        print "optimizing " + attribute

        for i in range(runs):

            print "value range: {0}".format(values)
            print "results    : {0}".format(results)


            for index in range(len(values)):
                current = float(values[index])

                if results[index] <= 0:                                         # only test if no result is available
                    self.set_attribute(attribute, current)

                    self.train()                                                # training

                    results[index] = max(self.net.trainingResults.validationAccuracy)
                                                                                # counting the best validation accuracy
                                                                                #   as training result


            # finding the best result and it's corresponding value and insert new values before and after

            bestIndex = np.argmax(results)
            print "best result in this run: {0}".format(results[bestIndex])


            if bestIndex == 0:                              # half current value if there is no lower one available
                lb = values[0] * 0.5

            else:                                           # interpolating between two values according to
                                                            #   their respective accuracies
                dist = abs(values[bestIndex] - values[bestIndex - 1])/2
                grad = 1 - 0.5*(results[bestIndex] - results[bestIndex - 1])/results[bestIndex]
                lb = values[bestIndex] - dist * grad


            if bestIndex >= len(values) - 1:                # +50% on current value if there is no larger one available
                ub = values[-1] * 1.5

            else:                                           # interpolating again
                dist = abs(values[bestIndex] - values[bestIndex + 1])/2
                grad = 1 - (results[bestIndex] - results[bestIndex + 1])/results[bestIndex]
                ub = values[bestIndex] + dist * grad


            results = np.insert(results, bestIndex+1, 0)
            results = np.insert(results, bestIndex,   0)
            values  = np.insert(values,  bestIndex+1, ub)
            values  = np.insert(values,  bestIndex,   lb)



        # selecting the best performing value for the network

        bestIndex = np.argmax(results)
        self.set_attribute(attribute, float(values[bestIndex]))
        self.net.trainingResults.validationAccuracy.append(float(results[bestIndex]))

        return


    def get_scales(self, attribute):

        if attribute == "eta":
            return [0.3, 1, 3]
        if attribute == "lambda":
            return [0.1, 1, 5]
        if attribute == "my":
            return [0.5, 1, 5]
        if attribute == "dropout":
            return [0.5, 1, 3]
        if attribute == "mini_batch_size":
            return [0.5, 1, 3]

    def get_attribute(self, attribute):

        if attribute == "eta":
            return self.eta
        if attribute == "lambda":
            return self.lmbda
        if attribute == "my":
            return self.my
        if attribute == "dropout":
            return self.dropout
        if attribute == "mini_batch_size":
            return self.mini_batch_size

    def set_attribute(self, attribute, value):

        if attribute == "eta":
            self.eta = value
            return

        if attribute == "lambda":
            self.lmbda = value
            return

        if attribute == "my":
            self.my = value
            return

        if attribute == "dropout":
            self.dropout = value
            return

        if attribute == "mini_batch_size":
            self.mini_batch_size = int(math.floor(value))
            return


def getLayers(mini_batch_size, dropout=0.0):

    feature_maps_1 = 5
    feature_maps_2 = 20
    n1 = 50

    return [
        ConvPoolLayer(image_shape=(mini_batch_size, 3, 32, 32),
                      filter_shape=(feature_maps_1, 3, 6, 6),
                      poolsize=(1, 1)),
        ConvPoolLayer(image_shape=(mini_batch_size, feature_maps_1, 27, 27),
                      filter_shape=(feature_maps_2, feature_maps_1, 6, 6),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=feature_maps_2*11*11, n_out=n1, activation_fn_name="ReLU", p_dropout=dropout),
        SoftmaxLayer(n_in=n1, n_out=43)]



class TrainingResults(object):
    """

    To contain relevant data from training

    """
    def __init__(self):
        self.trainingCost = []
        self.trainingAccuracy = []
        self.validationCost = []
        self.validationAccuracy = []

        self.fivePointAverageImprovement = []

        self.testAccuracy = 0.0
        self.testCost = 0.0

        self.trainingSize = 0
        self.testSize = 0
        self.validationSize = 0

        self.trainingTime = 0.0



class CsvWriter(object):
    """

    csv logger class

    """
    def __init__(self, path, delimiter=";"):
        self.path = path
        self.delimiter = delimiter
        self.opened = False
#        self.open()

    def write(self, data):
        if not self.opened:
            self.open()
        self.writer.writerows([data])

    def open(self):
        self.opened = True
        self.file = open(self.path, "wb")
        self.writer = csv.writer(self.file, delimiter=self.delimiter)

    def close(self):
        self.opened = False
        self.file.close()



class Network(object):
    """

    Main network class

    """

    def __init__(self, layers, writer, mini_batch_size, name):
        """

        creating a new network instance using the provided layers

        """
        self.writer = writer
        self.layers = layers
        self.name = name
        self.weights = []
        self.velocities = []
        self.biases = []

        for layer in self.layers:
            self.weights.append(layer.w)
            self.velocities.append(layer.v)
            self.biases.append(layer.b)

        self.x = T.matrix("x")
        self.y = T.ivector("y")

        self.set_mini_batch_size(mini_batch_size)

        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.trainingResults = TrainingResults()


    def set_mini_batch_size(self, new_batch_size):
        """

        allows reshaping the layers for different mini batch sizes, especially useful for reusing networks

        """

        self.mini_batch_size = new_batch_size

        init_layer = self.layers[0]

        if isinstance(init_layer, ConvPoolLayer):
            init_layer.set_mini_batch_size(self.mini_batch_size)

        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]

            if isinstance(layer, ConvPoolLayer):
                layer.set_mini_batch_size(self.mini_batch_size)

            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)



    def MBSGD(self, training_data, validation_data, test_data, max_epochs, eta, lmbda, my, early_stopping):
        """

        actual training method, performs momentum based stochastic gradient descent

        """
        modified_eta = eta

        self.writer.write(["Training Data"])
        self.writer.write([size(training_data)])
        self.writer.write(["Validation Data"])
        self.writer.write([size(test_data)])
        self.writer.write(["Test Data"])
        self.writer.write([size(validation_data)])
        self.writer.write(["Batch-Size"])
        self.writer.write([self.mini_batch_size])
        self.writer.write(["Epochs Max"])
        self.writer.write([max_epochs])
        self.writer.write(["Lambda"])
        self.writer.write([lmbda])
        self.writer.write(["Momentum"])
        self.writer.write([my])
        self.writer.write(["Dropout"])
        self.writer.write([self.layers[-1].p_dropout])
        self.writer.write(["CPU"])

        self.writer.write([""])
        self.writer.write(["Epoch", "Eta", "Validation Acc", "Test Acc", "5-Point Avg. Slope"])


        """Train the network using momentum based stochastic gradient descent."""

        training_x,     training_y      = training_data
        validation_x,   validation_y    = validation_data
        test_x,         test_y          = test_data


        # compute number of minibatches for training, validation and testing
        num_tra_batches = size(training_data)   /self.mini_batch_size
        num_val_batches = size(validation_data) /self.mini_batch_size
        num_tes_batches = size(test_data)       /self.mini_batch_size


        # returns the sum of all the squared weights in the network
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])



        # define the (regularized) cost function, symbolic gradients, and updates

        cost = self.layers[-1].cost(self) + 0.5*lmbda*l2_norm_squared/num_tra_batches

        gradsW = T.grad(cost, self.weights)
        gradsB = T.grad(cost, self.biases)

        updatesW = []
        updatesV = []
        for weight, vel, gradW in zip(self.weights, self.velocities, gradsW):
            new_vel = my*vel - modified_eta*gradW
            updatesW.append((weight, weight + new_vel))
            updatesV.append((vel, new_vel))

        updatesB = [(bias, bias-modified_eta*gradB) for bias, gradB in zip(self.biases, gradsB)]

        updates = updatesW + updatesV + updatesB

        # define functions to train a mini-batch, and to compute the accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index

        train_mb = theano.function([i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        validate_mb_accuracy = theano.function([i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        test_mb_accuracy = theano.function([i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        self.test_mb_predictions = theano.function([i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        initial_val_accuracy = np.mean([validate_mb_accuracy(j) for j in xrange(num_val_batches)])
        print "initial: validation accuracy {0:.2%}".format(initial_val_accuracy)

        # Do the actual training
        best_validation_accuracy = 0.0
        test_accuracy = 0.0
        accuracy_slope = []
        last_accuracy = 0.0

        start = time.time()

        for epoch in xrange(max_epochs):

            batch_cost = []

            training_start = time.time()

            for minibatch_index in xrange(num_tra_batches):

                cost_ij = train_mb(minibatch_index)
                batch_cost.append(cost_ij)

            training_duration = time.time() - training_start

            if epoch % 5 == 0 or epoch == max_epochs-1:

                validation_accuracy = np.mean([validate_mb_accuracy(j) for j in xrange(num_val_batches)])

                self.trainingResults.validationAccuracy.append(validation_accuracy)
                self.trainingResults.validationCost.append(np.average(batch_cost))

                accuracy_slope.append(validation_accuracy - last_accuracy)
                if len(accuracy_slope) > 5: accuracy_slope = accuracy_slope[1:]
                last_accuracy = validation_accuracy

                five_point_average_slope = np.average(accuracy_slope)
                self.trainingResults.fivePointAverageImprovement.append(five_point_average_slope)

                isBest = ""

                if validation_accuracy >= best_validation_accuracy:
                    isBest = " (best)"
                    best_validation_accuracy = validation_accuracy

                    test_accuracy = np.mean([test_mb_accuracy(j) for j in xrange(num_tes_batches)])

                    self.trainingResults.testAccuracy = test_accuracy
                    isBest = "{0}, test accuracy {1:.2%}".format(isBest, test_accuracy)

                self.save(self.name)

                self.writer.write([epoch,
                                   modified_eta,
                                   validation_accuracy,
                                   test_accuracy,
                                   five_point_average_slope])



                print("Epoch {0}: validation accuracy {1:.2%}{2}".format(epoch, validation_accuracy, isBest))

                if early_stopping and (epoch > 4 and abs(five_point_average_slope) < 0.05):
                    print "training stopped"
                    break

                ### !!!!!!!!!!!!
                ### reducing eta over epochs does not work in this version. problems with theano shared variables
                ### !!!!!!!!!!!!

                if modified_eta > 0.01:
                    modified_eta = modified_eta/1.8
                else:
                    if modified_eta > 0.00001:
                        modified_eta = modified_eta / 1.4
                    else:
                        if modified_eta > 0.0000001:
                            modified_eta = modified_eta / 1.2


        self.trainingResults.trainingTime = time.time() - start
        self.writer.write(["Time", self.trainingResults.trainingTime])
        self.writer.write([""])
        self.writer.write([""])

        print("Finished training. Best validation accuracy: {0:.2%}, test accuracy: {1:.2%}".format(best_validation_accuracy, test_accuracy))


    def evaluate(self, data):
        """

        classify the given data and return the obtained accuracy

        """

        test_x, test_y = data
        num_batches = size(data) / self.mini_batch_size

        i = T.lscalar()  # mini-batch index

        test_mb_accuracy = theano.function([i], self.layers[-1].accuracy(self.y),
                                           givens={
                                               self.x:
                                                   test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                                               self.y:
                                                   test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
                                           })

        return np.mean([test_mb_accuracy(j) for j in xrange(num_batches)])


    def classify(self, data):
        """

        classify the given data and return classification vector

        """

        test_x,         test_y          = data
        num_batches = size(data)       /self.mini_batch_size

        i = T.lscalar() # mini-batch index

        test_mb_classification = theano.function([i], self.layers[-1].classification(),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        classes = []
        for j in xrange(num_batches):
            for result in test_mb_classification(j):
                classes.append(result)

        return classes



    def save(self, filename=None):
        """

        Save the network to disk

        """

        if filename is None: filename = self.name

        data = {"layers": [layer.to_dict() for layer in self.layers],
                "mbs": self.mini_batch_size}

        f = open(filename, "wb")
        cPickle.dump(data, f)
        f.close()



def load(filename):
    """

    load a network from disk, returning a new instance

    """

    f = open(filename, "rb")
    data = cPickle.load(f)
    f.close()

    layers = [layer_factory(dictL) for dictL in data["layers"]]

    net = Network(layers, writer=None, mini_batch_size=data["mbs"], name=filename)

    print "loaded net: {0}".format(filename)

    return net



def layer_factory(dictionary):
    """

    create layers from the provided data, used for loading networks

    """

    if dictionary["type"] == "conv":
        return ConvPoolLayer((0, 0), (0, 0), dictionary=dictionary)

    if dictionary["type"] == "full":
        return FullyConnectedLayer(0, 0, dictionary=dictionary)

    if dictionary["type"] == "soft":
        return SoftmaxLayer(0, 0, dictionary=dictionary)



# layer types

class ConvPoolLayer(object):
    """

    a convolutional layer with max-pooling

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn_name="ReLU", dictionary=None):
        """

        `filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.


        when loading from dictionary, activation function is currently set to ReLu only

        """

        w, v, b = None, None, None

        if dictionary is not None:
            filter_shape = dictionary["filter_shape"]
            image_shape = dictionary["image_shape"]
            poolsize = dictionary["poolsize"]
            w = theano.shared(dictionary["w"])
            v = theano.shared(dictionary["v"])
            b = theano.shared(dictionary["b"])
            activation_fn_name = dictionary["activation_fn_name"]


        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn_name = activation_fn_name

        if self.activation_fn_name == "ReLU":
            self.activation_fn = ReLU
        else:
            if self.activation_fn_name == "sigmoid":
                self.activation_fn = sigmoid
            else:
                self.activation_fn = linear

        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))

        if w is None:
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                    dtype=theano.config.floatX),
                borrow=True)
        else:
            self.w = w
        if v is None:
            self.v = theano.shared(
                np.asarray(
                    np.zeros(shape=filter_shape),
                    dtype=theano.config.floatX),
                borrow=True)
        else:
            self.v = v
        if b is None:
            self.b = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                    dtype=theano.config.floatX),
                borrow=True)
        else:
            self.b = b

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv2d(input=self.inpt, filters=self.w, filter_shape=self.filter_shape, input_shape=self.image_shape)
        pooled_out = pool.pool_2d(
            input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

    def get_outpt(self):
        return self.output_dropout

    def set_mini_batch_size(self, mbs):
        a, b, c, d = self.image_shape
        self.image_shape = mbs, b, c, d

    def to_dict(self):
        w = self.w.get_value()
        v = self.v.get_value()
        b = self.b.get_value()

        data = {"filter_shape": self.filter_shape,
                "image_shape": self.image_shape,
                "poolsize": self.poolsize,
                "activation_fn_name": self.activation_fn_name,
                "w": self.w.get_value(),
                "v": self.v.get_value(),
                "b": self.b.get_value(),
                "type": "conv"}

        return data

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn_name="ReLU", p_dropout=0.0, dictionary=None):

        w, v, b = None, None, None


        if dictionary is not None:
            n_in = dictionary["n_in"]
            n_out = dictionary["n_out"]
            p_dropout = dictionary["p_dropout"]
            w = theano.shared(dictionary["w"])
            v = theano.shared(dictionary["v"])
            b = theano.shared(dictionary["b"])
            activation_fn_name = dictionary["activation_fn_name"]


        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.activation_fn_name = activation_fn_name

        if self.activation_fn_name == "ReLU":
            self.activation_fn = ReLU
        else:
            if self.activation_fn_name == "sigmoid":
                self.activation_fn = sigmoid
            else:
                self.activation_fn = linear


        # Initialize weights and biases
        if w is None:
            self.w = theano.shared(
                np.asarray(
                    np.random.normal(
                        loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                    dtype=theano.config.floatX),
                name='w', borrow=True)
        else:
            self.w = w

        if v is None:
            self.v = theano.shared(
                np.asarray(
                    np.zeros(shape=(n_in, n_out)),
                    dtype=theano.config.floatX),
                name='v', borrow=True)
        else:
            self.v = v

        if b is None:
            self.b = theano.shared(
                np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                           dtype=theano.config.floatX),
                name='b', borrow=True)
        else:
            self.b = b

        self.params = [self.w, self.v, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def to_dict(self):
        data = {"n_in": self.n_in,
                "n_out": self.n_out,
                "p_dropout": self.p_dropout,
                "activation_fn_name": self.activation_fn_name,
                "w": self.w.get_value(),
                "v": self.v.get_value(),
                "b": self.b.get_value(),
                "type": "full"}

        return data

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0, dictionary=None):

        w, v, b = None, None, None

        if dictionary is not None:
            n_in = dictionary["n_in"]
            n_out = dictionary["n_out"]
            p_dropout = dictionary["p_dropout"]
            w = theano.shared(dictionary["w"])
            v = theano.shared(dictionary["v"])
            b = theano.shared(dictionary["b"])


        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout

        # Initialize weights and biases
        if w is None:
            self.w = theano.shared(
                np.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='w', borrow=True)
        else:
            self.w = w

        if v is None:
            self.v = theano.shared(
                np.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='v', borrow=True)
        else:
            self.v = v

        if b is None:
            self.b = theano.shared(
                np.zeros((n_out,), dtype=theano.config.floatX),
                name='b', borrow=True)
        else:
            self.b = b

        self.params = [self.w, self.v, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):

        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)

        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        """

        Return the log-likelihood cost

        """
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def classification(self):

        return self.output

    def accuracy(self, y):
        """

        Return the accuracy for the mini-batch.

        """
        return T.mean(T.eq(y, self.y_out))


    def to_dict(self):
        data = {"n_in": self.n_in,
                "n_out": self.n_out,
                "p_dropout": self.p_dropout,
                "w": self.w.get_value(),
                "v": self.v.get_value(),
                "b": self.b.get_value(),
                "type": "soft"}

        return data


# other stuff

def size(data):
    """

    Return the size of the dataset `data`

    """

    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)


