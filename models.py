
import lasagne
import theano
import theano.tensor as T
import text_utils
import pickle as pkl
import utils
import numpy as np


class Model(object):
    def __init__(self, options, it=None):
        self.options = options
        self.it = it

    def initialise(self):

        network_fn = self.build_network()
        self.network = network_fn()

    def reload(self, model_file):

        # Reloading
        options = pkl.load(open(model_file + '.pkl'))

        self.options = options
        network_fn = self.build_network()
        network = network_fn()
        print "reloading {}...".format(model_file)
        self.network = utils.load_model(network, model_file)

    def build_network(self):
        raise NotImplementedError

    def get_loss(self, prediction, target):
        loss = lasagne.objectives.squared_error(prediction, target)
        loss = loss.mean()
        return loss




    def compile_theano_func(self, lr):
        
        # Prepare Theano variables for inputs and targets
        inputs, input_vars = self._get_inputs()
        target = T.tensor4('targets')

        target_var = target.dimshuffle((0, 3, 1, 2))

        #get for prediction
        prediction = lasagne.layers.get_output(self.network, *input_vars)

        #get our prediction
        loss = self.get_loss(prediction, target_var)

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(
            loss, params, learning_rate=lr)

        test_prediction = lasagne.layers.get_output(self.network, *input_vars, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
        test_loss = test_loss.mean()

        print "Computing the functions..."
        self.train_fn = theano.function(inputs + [target], [loss, prediction.transpose((0, 2, 3, 1))], updates=updates,
                                        allow_input_downcast=True)

        # Compile a second function computing the validation loss and accuracy:
        self.val_fn = theano.function(inputs + [target], [test_loss, test_prediction.transpose((0, 2, 3, 1))],
                                      allow_input_downcast=True)

    def _get_inputs(self):
        input = T.tensor4('inputs')
        input_var = input.transpose((0, 3, 1, 2))
        return [input], [input_var]

    def train(self, imgs, target, caps):

        res = self.train_fn(imgs, target)
        return res

    def get_generation_fn(self):

        def val_fn(imgs, target, caps):
            res = self.val_fn(imgs, target)
            return res


        return val_fn




class cnn_EA(Model):

    def build_network(self):
        def build_basic_cnn(input_var=None):

            num_units = self.options['num_units']
            encoder_size = self.options['encoder_size']

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
            network = lasagne.layers.Conv2DLayer( network, num_filters=encoder_size, filter_size=(6, 6))

            # Deconv
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
        def build_basic_cnn_v2(input_var=None):

            num_units = self.options['num_units']
            encoder_size = self.options['encoder_size']

            print "We have {} hidden units".format(num_units)

            network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                                input_var=input_var)

            network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_units, filter_size=(5, 5))
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
            network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_units, filter_size=(3, 3))
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_units, filter_size=(3, 3))
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            # hidden units
            network = lasagne.layers.Conv2DLayer(network, num_filters=encoder_size, filter_size=(6, 6), nonlinearity=lasagne.nonlinearities.tanh)

            # Deconv
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(6, 6), stride=(1, 1))
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(5, 5), stride=(2,2))
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(4, 4), stride=(2,2), nonlinearity=lambda x: x.clip(0., 1.))

            return network

        return build_basic_cnn_v2

class cnn_v2_sharp(cnn_v2):

    def get_loss(self, prediction, target):

        lb_std = self.options['lb_std']

        loss = lasagne.objectives.squared_error(prediction, target)
        loss = loss.mean() - lb_std * theano.tensor.std(prediction, axis=(1, 2, 3)).mean()
        return loss


class cnn_v3(Model):

    def build_network(self):
        def build_basic_cnn_v2(input_var=None):

            num_units = self.options['num_units']
            encoder_size = self.options['encoder_size']

            print "We have {} hidden units".format(num_units)

            network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                                input_var=input_var)

            network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_units, filter_size=(5, 5))
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
            network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_units, filter_size=(3, 3))
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_units, filter_size=(3, 3))
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            # fully connected
            network = lasagne.layers.FlattenLayer(network)
            network = lasagne.layers.DenseLayer(network, num_units*6*6)
            network = lasagne.layers.DenseLayer(network, encoder_size)
            network = lasagne.layers.DenseLayer(network, num_units*6*6)
            network = lasagne.layers.ReshapeLayer(network, (input_var.shape[0], num_units, 6, 6))

                                        # Deconv
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(16, 16), stride=(1,1))
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(8, 8), stride=(1,1))
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(3, 3), stride=(1,1))
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(3, 3), stride=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid)

            return network

        return build_basic_cnn_v2
    
class cnn_batchnorm(Model):

    def build_network(self):
        def build_basic_cnn_v2(input_var=None):

            num_units = self.options['num_units']
            encoder_size = self.options['encoder_size']

            print "We have {} hidden units".format(num_units)

            network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                                input_var=input_var)
            lasagne.layers.BatchNormLayer(network)

            network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_units, filter_size=(5, 5))
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
            lasagne.layers.BatchNormLayer(network)
            network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_units, filter_size=(3, 3))
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
            
            lasagne.layers.BatchNormLayer(network)
            network = lasagne.layers.Conv2DLayer(
                network, num_filters=num_units, filter_size=(3, 3))
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

            # fully connected
            lasagne.layers.BatchNormLayer(network)
            network = lasagne.layers.FlattenLayer(network)
            network = lasagne.layers.DenseLayer(network, num_units*6*6)
            lasagne.layers.BatchNormLayer(network)
            network = lasagne.layers.DenseLayer(network, encoder_size)
            lasagne.layers.BatchNormLayer(network)
            network = lasagne.layers.DenseLayer(network, num_units*6*6)
            network = lasagne.layers.ReshapeLayer(network, (-1, num_units, 6, 6))

                                        # Deconv
            lasagne.layers.BatchNormLayer(network)
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(16, 16), stride=(1,1))
            lasagne.layers.BatchNormLayer(network)
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(8, 8), stride=(1,1))
            lasagne.layers.BatchNormLayer(network)
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(3, 3), stride=(1,1))
            lasagne.layers.BatchNormLayer(network)
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(3, 3), stride=(1,1), nonlinearity=lasagne.nonlinearities.sigmoid)

            return network

        return build_basic_cnn_v2
    
    def get_loss(self, prediction, target):

        lb_std = self.options['lb_std']

        loss = lasagne.objectives.squared_error(prediction, target)
        loss = loss.mean() - lb_std * theano.tensor.std(prediction, axis=(1, 2, 3)).mean()
        return loss
    
class caps_model(Model):

    def __init__(self, options, it = None):
        super(caps_model, self).__init__(options, it)
        self.W = None

    def _get_inputs(self):
        input = T.imatrix('captions')
        rval = input

        use_bag_of_word = self.options['use_bag_of_word']
        if use_bag_of_word:
            #Get the embeddings and sum.
            W = self.get_emb()
            rval = W[input]
            rval = rval.sum(axis=1)

        return [input], [rval]

    def train(self, imgs, target, caps):


        #We only keep the first caps for now
        caps = [cap[np.random.choice(len(cap))] for cap in caps]
        caps = text_utils.pad_to_the_max(caps)

        res = self.train_fn(caps, target)
        return res

    def get_generation_fn(self):

        def val_fn(imgs, target, caps):

            # We only keep the first caps for now
            caps = [cap[0] for cap in caps]
            caps = text_utils.pad_to_the_max(caps)

            res = self.val_fn(caps, target)
            return res

        return val_fn

    def get_emb(self):


        if self.W is None:
            vocab_size = self.options['vocab_size']
            emb_file = self.options['emb_file']
            emb_size = self.options['emb_size']

            self.W = np.random.normal(loc=0.0, scale=0.01, size=(vocab_size, emb_size)).astype('float32')


            # Loading the embeding file.
            if emb_file is not None:

                print "We have pretrained embedings."
                it = self.it

                nb_present = 0
                for i, line in enumerate(open(emb_file)):
                    line = line.split(' ')
                    word = line[0]
                    emb = [float(x) for x in line[1:]]

                    if word in it.vocab:
                        self.W[it.mapping[word]] = emb
                        nb_present += 1

                    if i % 100000 == 0:
                        print "Done {} words".format(i)

                print "There were {} on {} words present.".format(nb_present, len(it.vocab))

            self.W = theano.shared(self.W)

        return self.W


    def build_network(self):
        def fn(input_var=None):

            num_units = self.options['num_units']
            emb_size = self.options['emb_size']
            vocab_size = self.options['vocab_size']
            rnn_size = self.options['rnn_size']
            use_bag_of_word = self.options['use_bag_of_word']

            print "We have {} hidden units".format(num_units)

            if use_bag_of_word:
                print "Using a neural bag of words."
                network = lasagne.layers.InputLayer(shape=(None, emb_size), input_var=input_var)
                network = lasagne.layers.DenseLayer(network, rnn_size)

            else:
                print "Using a recurrent nnet."
                W = self.get_emb()
                network = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=input_var)
                network = lasagne.layers.EmbeddingLayer(network, vocab_size, emb_size, W=W)
                network = lasagne.layers.GRULayer(network, rnn_size, only_return_final=True)


            # fully connected
            network = lasagne.layers.ReshapeLayer(network, (-1, rnn_size, 1, 1))

            # Deconv
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units*2, filter_size=(7, 7), stride=(1, 1))
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(4, 4), stride=(2,2))
            network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(2, 2), stride=(2,2), nonlinearity=lambda x: x.clip(0., 1.))


            return network

        return fn

    def get_loss(self, prediction, target):

        lb_std = self.options['lb_std']

        loss = lasagne.objectives.squared_error(prediction, target)
        loss = loss.mean() - lb_std * theano.tensor.std(prediction, axis=(1, 2, 3)).mean()
        return loss