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

def load_dataset(batch_size=128, load_caption=False):

    train_iter = iterator.PreprocessIterator(batch_size=batch_size, extract_center=True, load_caption=load_caption)
    val_iter = iterator.Iterator(nb_sub=2000, batch_size=batch_size, img_path = 'val2014', extract_center=True, load_caption=load_caption)

    #val_iter.caption_dict = train_iter.caption_dict
    val_iter.vocab = train_iter.vocab
    val_iter.mapping = train_iter.mapping
    val_iter.process_captions()

    return train_iter, val_iter


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train(network_cl, num_epochs=20,
          lr=0.001, sample=3, save_freq=100,
          batch_size=128, verbose_freq=100,
          model_file="models/testing123.npz",
          reload=False,
          load_caption = False,
          **kwargs):

    # Load the dataset
    print "Loading data..."
    train_iter, val_iter = load_dataset(batch_size, load_caption=load_caption)



    #some monitoring stuff
    val_loss = []
    train_loss = []

    my_model = network_cl(kwargs, train_iter)
    # Reloading
    if reload:
        my_model.reload(model_file)
    else:
        my_model.initialise()

    my_model.compile_theano_func(lr=lr)

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
            train_err_tmp, pred = my_model.train(inputs, targets, caps)
            train_err += train_err_tmp
            train_batches += 1


            # Generate
            if (i+1) % verbose_freq == 0.:
                utils.generate_and_show_sample(my_model.get_generation_fn(), nb=sample, seed=-1, it=train_iter)
                print "batch {} of epoch {} of {} took {:.3f}s".format(i, epoch + 1, num_epochs, time.time() - start_time)
                print "  training loss:\t\t{:.6f}".format(train_err / train_batches)

            if (i+1) % save_freq == 0:
                print "saving the model"
                utils.save_model(my_model.network, kwargs, model_file)

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

    return my_model



if __name__ == '__main__':

    val_fn, network = train(build_cnn, 1)
    utils.show_sample(val_fn, nb=3)