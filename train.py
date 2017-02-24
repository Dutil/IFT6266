import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import utils
import lasagne
import iterator
import pickle as pkl


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset(batch_size=128):

    train_iter = iterator.Iterator(batch_size=batch_size, extract_center=True)
    val_iter = iterator.Iterator(nb_sub=2000, batch_size=batch_size, img_path = 'val2014', extract_center=True)

    return train_iter, val_iter


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train(network_fn, num_epochs=20,
          lr=0.001, sample=3, save_freq=100,
          batch_size=128, verbose_freq=100,
          model_file="models/testing123.npz",
          reload=False,
          **kwargs):

    # Load the dataset
    print "Loading data..."
    train_iter, val_iter = load_dataset(batch_size)

    #some monitoring stuff
    val_loss = []
    train_loss = []

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.tensor4('targets')

    input_var = input.transpose((0, 3, 1, 2))
    target_var = target.dimshuffle((0, 3, 1, 2))

    # Reloading
    if reload:
        options = pkl.load(open(model_file+'.pkl'))
        kwargs = options

    network = network_fn(input_var, **kwargs)
    if reload:
        print "reloading {}...".format(model_file)
        network = utils.load_model(network, model_file)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(
            loss, params, learning_rate=lr)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    print "Computing the functions..."
    #prediction = prediction.transpose((0, 2, 3, 1))
    train_fn = theano.function([input, target], [loss, prediction.transpose((0, 2, 3, 1))], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_prediction.transpose((0, 2, 3, 1))])

    # Finally, launch the training loop.
    print "Starting training..."
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i, batch in enumerate(train_iter):
            inputs, targets, caps = batch
            train_err_tmp, pred = train_fn(inputs, targets)
            train_err += train_err_tmp
            train_batches += 1


            # Generate
            if (i+1) % verbose_freq == 0.:
                utils.generate_and_show_sample(val_fn, nb=sample)
                print "batch {} of epoch {} of {} took {:.3f}s".format(i, epoch + 1, num_epochs, time.time() - start_time)
                print "  training loss:\t\t{:.6f}".format(train_err / train_batches)

            if (i+1) % save_freq == 0:
                print "saving the model"
                utils.save_model(network, kwargs, model_file)

        train_loss.append(train_err)

        # And a full pass over the validation data:
        #val_err = 0
        #val_batches = 0

        #for batch in val_iter:
        #    inputs, targets, caps = batch
        #    err, pred = val_fn(inputs, targets)
        #    val_err += err
        #    val_batches += 1

        # Then we print the results for this epoch:

        #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))



    #import ipdb
    #ipdb.set_trace()

    return val_fn, network



if __name__ == '__main__':

    val_fn, network = train(build_cnn, 1)
    utils.show_sample(val_fn, nb=3)