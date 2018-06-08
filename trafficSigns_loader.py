import random
import math
import cPickle
import gzip
import theano
import theano.tensor as T

# Third-party libraries
import numpy as np

import imagePrepare

def shared(data):
    """

    Place the data into shared variables.  This allows Theano to copy
    the data to the GPU, if one is available.

    """
    shared_x = theano.shared(
        np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, "int32")


def load_data_from_images(load_training=True, load_validation=True, load_test=True, smallSet = False,
                          obstructions = False, contrast = False, sharpness = False, translations = False):

    """

    load and preprocess the provided images, data expansion is optional and only applied to training data

    """


    if smallSet:                    # limit loaded data to save time during development
        limitTrVal = 4000
        limitTest = 1200
        print "loading small set"
    else:
        limitTrVal = 0
        limitTest = 0
        print "loading large set"



    if load_test:
        test = imagePrepare.loadImageData("./recognition/data/GTSRB_Test/Final_Test/Images/", threshold=limitTest)
    else:
        test = ([0], [0], [0])

    if load_training and not load_validation:
        training = imagePrepare.loadImageData("./recognition/data/GTSRB_Training/Final_Training/Images/",
                                              threshold=limitTrVal, includeValidation=False,
                                              obstructions=obstructions, contrast=contrast, sharpness=sharpness, translations=translations)
        validation = ([0], [0], [0])
    else:
        if load_training and load_validation:
            training, validation = imagePrepare.loadImageData("./recognition/data/GTSRB_Training/Final_Training/Images/",
                                              threshold=limitTrVal, includeValidation=True,
                                              obstructions=obstructions, contrast=contrast, sharpness=sharpness, translations=translations)
        else:
            training = ([0], [0], [0])
            validation = ([0], [0], [0])


    return [shared(training), shared(validation), shared(test), training[2], validation[2], test[2]]






