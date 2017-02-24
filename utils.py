

from iterator import Iterator
import matplotlib.pyplot as plt
import numpy as np
import lasagne
import theano
from theano import tensor as T
import pickle as pkl


def generate_and_show_sample(fn, nb=1, random=True):

    it = Iterator(img_path="val2014")
    choice = range(len(it))
    if random == True:
        np.random.shuffle(choice)

    choice = choice[:nb]

    try:
        xs, ys, cs = zip(*[it[i] for i in choice])
        loss, preds = fn(xs, ys)

        show_sample(xs, ys, preds, nb)
    except:
        print "Oups!"
        
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

def show_sample(xs, ys, preds, nb=1):

    for i in range(nb):
        img_true = np.copy(xs[i])
        center = (int(np.floor(img_true.shape[0] / 2.)), int(np.floor(img_true.shape[1] / 2.)))

        img_true[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = ys[i]

        plt.subplot(2, nb, i+1)
        plt.imshow(img_true)

        img_pred = np.copy(xs[i])
        img_pred[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = preds[i]
        plt.subplot(2, nb, nb+i+1)
        plt.imshow(img_pred)

    plt.show()



def save_model(network, options, file_name):
    np.savez(file_name, *lasagne.layers.get_all_param_values(network))
    option_file = file_name + '.pkl'
    pkl.dump(options, open(option_file, 'w'))



def load_model(network, file_name):
    with np.load(file_name) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    return network