

from iterator import Iterator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import lasagne
import theano
from theano import tensor as T
import pickle as pkl
import os

def put_in_middle(img, middle):
    
    center = (int(np.floor(img.shape[1] / 2.)), int(np.floor(img.shape[2] / 2.)))
    img = np.copy(img)
    img[:, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = middle
    return img

def put_in_middle_theano(img, middle):
    center = (img.shape[2] / 2, img.shape[3] / 2)

    img = T.set_subtensor(img[:, :, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16],
                              middle)

    return img

def generate_and_show_sample(fn, nb=1, seed=1993, it=None, verbose=True, n_split=1, return_64_64=False, replace_middle=False):

    if it is None:
        it = Iterator(img_path="val2014", load_caption=False, process_text=True)

    choice = range(len(it))
    if seed > 0:
        np.random.seed(seed)
        np.random.shuffle(choice)

    choice = choice[:nb] * 5

    #try:
    xs, ys, cs = zip(*[it[i] for i in choice])
    loss, preds = fn(xs, ys, cs)

    figs = []

    for pl in np.array_split(np.arange(nb), n_split):
            figs.append(show_sample([xs[i] for i in pl], [ys[i] for i in pl], [preds[i] for i in pl], len(pl),
                        return_64_64=return_64_64, replace_middle=replace_middle))
    #except Exception as e:
    #    print e
    #    print "Oups!"

    caps = []

    try:
        if verbose and it.mapping is not None:
            for img in cs:
                sentence = [it.mapping[idx] for idx in img[0]]
                caps.append(' '.join(sentence))

    except AttributeError:
        pass

    return figs, caps

def get_theano_generative_func(network_path, network_fn):



    input = T.tensor4('inputs')
    target = T.tensor4('targets')

    input_var = input.transpose((0, 3, 1, 2))
    target_var = target.dimshuffle((0, 3, 1, 2))

    network = network_fn(input_var)
    network = load_model(network, network_path)

    test_prediction = lasagne.layers.get_output(network, input_var, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    print "Computing the functions..."
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_prediction.transpose((0, 2, 3, 1))])
    return val_fn

def show_sample(xs, ys, preds, nb=1, return_64_64=False, replace_middle=False):

    fig = plt.figure()
    gcf = plt.gcf()
    gcf.set_size_inches(18, 15)
    fig.set_canvas(gcf.canvas)

    for i in range(nb):
        img_true = np.copy(xs[i])
        center = (int(np.floor(img_true.shape[0] / 2.)), int(np.floor(img_true.shape[1] / 2.)))

        img_true[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = ys[i]

        ax = fig.add_subplot(2, nb, i+1)
        ax.imshow(img_true)


        # plt.imshow(img_true)
        #fig.subplot(2, nb, i+1)

        if not return_64_64:
            img_pred = np.copy(xs[i])
            #print preds[i].shape
            img_pred[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = preds[i]

            ax = fig.add_subplot(2, nb, nb+i+1)
            ax.imshow(img_pred)

            #fig.subplot(2, nb, nb+i+1)
            #fig.imshow(img_pred)
        else:
            
            if replace_middle == False:
                ax = fig.add_subplot(2, nb, nb+i+1)
                ax.imshow(preds[i])
            else:
                img_pred = np.copy(xs[i])
                img_pred[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = preds[i][center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
                
                ax = fig.add_subplot(2, nb, nb+i+1)
                ax.imshow(img_pred)
            #fig.subplot(2, nb, nb+i+1)
            #fig.imshow(preds[i])
            
    return fig



def save_model(network, options, file_name):
    np.savez(file_name, *lasagne.layers.get_all_param_values(network))
    option_file = file_name + '.pkl'
    pkl.dump(options, open(option_file, 'w'))



def load_model(network, file_name):
    with np.load(file_name) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    return network