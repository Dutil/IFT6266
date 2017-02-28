
import lasagne
import theano
import theano.tensor as T


class Model(object):
    def __init__(self):
        pass

    def build_network(self):
        raise NotImplementedError

    def get_loss(self, prediction, target):
        loss = lasagne.objectives.squared_error(prediction, target)
        loss = loss.mean()
        return loss

class cnn_EA(Model):

    def build_network(self):
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
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(5, 5), nonlinearity=None)
            network = lasagne.layers.Upscale2DLayer(network, 2)
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(5, 5), nonlinearity=None)
            #network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(8, 8), stride=(2, 2))

            return network

        return build_basic_cnn



class cnn_v2(Model):

    def build_network(self):
        def build_basic_cnn_v2(input_var=None, num_units=32, encoder_size=100):

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
            network = lasagne.layers.Conv2DLayer(network, num_filters=encoder_size, filter_size=(6, 6))

            # Deconv
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(6, 6), stride=(1, 1))
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(5, 5), stride=(2,2))
            #network = lasagne.layers.Upscale2DLayer(network, 2)
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(4, 4), stride=(2,2), nonlinearity=lambda x: x.clip(0., 1.))

            return network

        return build_basic_cnn_v2

class cnn_v2_sharp(cnn_v2):

    def get_loss(self, prediction, target):
        loss = lasagne.objectives.squared_error(prediction, target)
        loss = loss.mean() - 0.2 * theano.tensor.std(prediction, axis=(1, 2)).mean()
        return loss