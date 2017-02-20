
import lasagne


def build_basic_mlp(input_var=None, nb_layer=1, num_units=800, nonlinearity=lasagne.nonlinearities.rectify):

    print "We have {} hidden layers, {} hidden units".format(nb_layer, num_units)


    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                        input_var=input_var)
    network = lasagne.layers.BatchNormLayer(network)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    for i in range(nb_layer):
        network = lasagne.layers.DenseLayer(network, 800, nonlinearity=nonlinearity)

    network = lasagne.layers.DenseLayer(network, 3*16*16, nonlinearity=lasagne.nonlinearities.identity)
    network = lasagne.layers.ReshapeLayer(incoming=network, shape=(input_var.shape[0], 3, 16, 16))
    network = lasagne.layers.Upscale2DLayer(network, 2)
    return network

def build_basic_cnn(input_var=None, num_units=32, encoder_size=800):

    print "We have {} hidden units".format(num_units)


    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=num_units, filter_size=(5, 5),
        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=num_units, filter_size=(3, 3))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
        network, num_filters=num_units, filter_size=(3, 3))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # hidden units
    #network = lasagne.layers.FlattenLayer(network)

    #network = lasagne.layers.DenseLayer(network, encoder_size)
    #network = lasagne.layers.DenseLayer(network, num_units*6*6)
    #network = lasagne.layers.ReshapeLayer(network, (input_var.shape[0], num_units, 6, 6))
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer( network, num_filters=encoder_size, filter_size=(6, 6))
    #network = lasagne.layers.Conv2DLayer(network, num_filters=encoder_size, filter_size=(1, 1))

    # Deconv
    #network = lasagne.layers.Upscale2DLayer(network, 2)
    network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(5,5))
    network = lasagne.layers.Upscale2DLayer(network, 2)
    network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(5, 5), nonlinearity=None)
    network = lasagne.layers.Upscale2DLayer(network, 2)
    network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(5, 5), nonlinearity=None)
    #network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(8, 8), stride=(2, 2))

    return network