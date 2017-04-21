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

    try:
        val_iter.vocab = train_iter.vocab
        val_iter.mapping = train_iter.mapping
        val_iter.process_captions()
    except Exception as e:
        print "The vocab passing didn't worked!"
        print e

    return train_iter, val_iter


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train(network_cl, num_epochs=20,
          lr=0.001, sample=3, save_freq=100,
          batch_size=128, verbose_freq=100,
          model_dir="models/testin123/",
          reload=False,
          load_caption = False, return_64_64=False, show_imgs = False,
          **kwargs):

    # Load the dataset
    print "Loading data..."
    train_iter, val_iter = load_dataset(batch_size, load_caption=load_caption)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #some monitoring stuff
    val_loss = []
    train_loss = []

    my_model = network_cl(kwargs, train_iter)
    # Reloading
    if reload:
        my_model.reload(model_dir)
    else:
        my_model.initialise()

    my_model.compile_theano_func(lr=lr)

    # Finally, launch the training loop.
    print "Starting training..."
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print "Doing epoch", epoch +1
        
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i, batch in enumerate(train_iter):
            inputs, targets, caps = batch
            losses = my_model.train(inputs, targets, caps)
            #train_err += train_err_tmp
            train_batches += 1


            # Generate
            if (i+1) % verbose_freq == 0.:
                #Get the images
                figs = utils.generate_and_show_sample(my_model.get_generation_fn(), nb=sample, seed=i, it=val_iter, n_split=2, return_64_64=return_64_64)

                for fig_no, fig in enumerate(figs):
                    fig_name = os.path.join(model_dir, "epoch_{}_batch_{}_split_{}.jpg".format(epoch, i, fig_no))

                    #save it
                    fig.savefig(fig_name)
                    if show_imgs:
                        fig.show()

                print "batch {} of epoch {} of {} took {:.3f}s".format(i, epoch + 1, num_epochs, time.time() - start_time)
                #print "  training loss:\t\t{:.6f}".format(train_err / train_batches)
                print "losses:", losses

            if (i+1) % save_freq == 0:
                print "saving the model to", model_dir
                my_model.save(model_dir)
                print "losses:", losses

        train_loss.append(train_err)

    return my_model



if __name__ == '__main__':
    import GAN

    # num_units = 512
    # my_model = train(GAN.SkipConditionalGAN, lr=0.0002, num_epochs=100, sample=10,
    #                                save_freq=600, verbose_freq=1000, batch_size=128, reload=False,
    #                               model_dir="models/skipcond_512_l2_60/",
    #                                generator_num_units = num_units,
    #                    generator_encoder_size = num_units,
    #                    discriminator_num_units = num_units,
    #                    discriminator_encoder_size = num_units,
    #                  noise_size = 5, use_wgan= False,
    #                  l2_penalty = 60, gan_penalty=0.1,
    #                    return_64_64 = True,
    #                   )
    #
    num_units = 128

    my_model = train(GAN.CapGAN, lr=0.0002, num_epochs=100, sample=10,
                           save_freq=600, verbose_freq=1000, batch_size=128, reload=False,
                           model_dir="models/CapGAN/",
                           generator_num_units=num_units,
                           load_caption=True,
                           generator_encoder_size=num_units,
                           discriminator_num_units=num_units,
                           discriminator_encoder_size=num_units,
                           num_units=num_units,
                           emb_size=100, vocab_size=7574, rnn_size=num_units, use_bag_of_word=True,
                           emb_file=None,
                           #emb_file="glove.6B/glove.6B.100d.txt",
                           noise_size=5, use_wgan=False,
                           l2_penalty=60, gan_penalty=0.1,
                           return_64_64=True,
                           show_imgs=True
                           )