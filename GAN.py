import lasagne
import theano
import theano.tensor as T
import text_utils
import pickle as pkl
import utils
import numpy as np
import models


class GAN(models.Model):

    def __init__(self, options, it = None):
        super(GAN, self).__init__(options, it)
        self.schedule = 0.

    def initialise(self):
        #TODO: build generator, discriminator
        generator_fn, discriminator_fn = self.build_network()

        self.generator = generator_fn()
        self.discriminator = discriminator_fn()

    def reload(self, model_file):
        # Reloading
        #TODO save the generator and the discriminator
        pass

    def build_generator(self, input_var=None):
        num_units = self.options['generator_num_units']
        encoder_size = self.options['generator_encoder_size']

        print "We have {} hidden units".format(num_units)

        network = lasagne.layers.InputLayer(shape=(None, 100),
                                            input_var=input_var)

        network = lasagne.layers.ReshapeLayer(network, (-1, 100, 1, 1))
        network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(4, 4),
                                                       stride=(1, 1))

        network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units/2,
                                                       filter_size=(5, 5), stride=(2, 2), crop=1)

        network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units/4,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2)

        #network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3,
                                                       filter_size=(4, 4), stride=(2, 2), crop=2,
                                                       nonlinearity=lasagne.nonlinearities.sigmoid)

        #network = lasagne.layers.BatchNormLayer(network)
        #network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(4, 4), stride=(2, 2))

        #network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(2, 2), stride=(2, 2),
        #                                               nonlinearity=lambda x: x.clip(0., 1.))

        return network

    def build_discriminator(self, input_var=None):
        num_units = self.options['discriminator_num_units']
        encoder_size = self.options['discriminator_encoder_size']
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

        print "We have {} hidden units".format(num_units)

        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                            input_var=input_var)

        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units/4, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)

        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units/2, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)

        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)


        network = lasagne.layers.FlattenLayer(network)
        network = lasagne.layers.DenseLayer(network, 1,
                                             nonlinearity=lasagne.nonlinearities.sigmoid)

        return network

    def build_network(self):
       return self.build_generator, self.build_discriminator

    def get_loss(self, disc_real, disc_sample):

        #WGAN loss
        discriminator_loss = -(T.log(disc_real) + T.log(1.-disc_sample)).mean()
        generator_loss = -T.log(disc_sample).mean()

        return discriminator_loss, generator_loss

    def compile_theano_func(self, lr):
        # Prepare Theano variables for inputs and targets
        inputs, input_vars = self._get_inputs()
        origin_real_img = T.tensor4('real_img')

        real_img = origin_real_img.dimshuffle((0, 3, 1, 2))

        # Generator
        sample = lasagne.layers.get_output(self.generator, *input_vars)

        #Discriminator
        disc_real = lasagne.layers.get_output(self.discriminator, real_img)
        disc_sample = lasagne.layers.get_output(self.discriminator, sample)

        # Our loss
        discriminator_loss, generator_loss = self.get_loss(disc_real, disc_sample)

        # generator update
        generator_params = lasagne.layers.get_all_params(self.generator, trainable=True)
        generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=lr, beta1=0.5)

        #Discriminator update
        discriminator_params = lasagne.layers.get_all_params(self.discriminator, trainable=True)
        discriminator_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=lr, beta1=0.5)


        print "Computing the functions..."


        self.train_generator_fn = None
        self.train_discriminator_fn = None
        self.train_generator_fn = theano.function(inputs, [generator_loss],
                                                  updates=generator_updates,
                                                  allow_input_downcast=True)

        self.train_discriminator_fn = theano.function(inputs + [origin_real_img], [discriminator_loss],
                                                  updates=discriminator_updates,
                                                  allow_input_downcast=True)

        # Compile a second function computing the validation loss and accuracy:
        self.generate_sample_fn = theano.function(inputs, [sample.transpose((0, 2, 3, 1))],
                                      allow_input_downcast=True)

    def _get_inputs(self):

        #TODO add noise

        input = T.matrix('noise')
        input_var = input#input.transpose((0, 3, 1, 2))
        return [input], [input_var]

    def train(self, imgs, target, caps):

        #TODO: train discriminator then generator
        noise = np.random.normal(size=(len(imgs), 100))

        [disc_loss] = self.train_discriminator_fn(noise, target)
        self.schedule = (self.schedule+1) % 10
        gen_loss = 0.
        if True or self.schedule == 0:
            [gen_loss] = self.train_generator_fn(noise)

        return gen_loss, disc_loss
        #return [disc_loss, gen_loss]

    def get_generation_fn(self):
        def val_fn(imgs, target, caps):
            #noise = np.random.normal(size=(len(imgs), 100))
            noise = np.random.uniform(size=(len(imgs), 100))
            res = self.generate_sample_fn(noise)

            #from IPython.core.debugger import Tracer
            #Tracer()()

            #print res[1].shape
            #print res[2].shape
            return 0, res[0]

        return val_fn
    
class BICGAN(GAN):
        #Under cosntruction
    
        def build_generator(self, input_var=None):
        num_units = self.options['generator_num_units']
        encoder_size = self.options['generator_encoder_size']

        print "We have {} hidden units".format(num_units)
        
        #Our encoder
        encoder = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                            input_var=input_var)
        
        encoder = lasagne.layers.BatchNormLayer(network)
        encoder = lasagne.layers.Conv2DLayer(network, num_filters=num_units/4, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        encoder = lasagne.layers.BatchNormLayer(network)

        encoder = lasagne.layers.Conv2DLayer(network, num_filters=num_units/2, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        encoder = lasagne.layers.BatchNormLayer(network)

        encoder = lasagne.layers.Conv2DLayer(network, num_filters=num_units, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        encoder = lasagne.layers.BatchNormLayer(network)
        encoder = lasagne.layers.FlattenLayer(network)
        

        network = lasagne.layers.InputLayer(shape=(None, 100),
                                            input_var=input_var)

        network = lasagne.layers.ReshapeLayer(network, (-1, 100, 1, 1))
        network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(4, 4),
                                                       stride=(1, 1))

        network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units/2,
                                                       filter_size=(5, 5), stride=(2, 2), crop=1)

        network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units/4,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2)

        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3,
                                                       filter_size=(4, 4), stride=(2, 2), crop=2,
                                                       nonlinearity=lasagne.nonlinearities.sigmoid)


        return network

    def build_discriminator(self, input_var=None):
        num_units = self.options['discriminator_num_units']
        encoder_size = self.options['discriminator_encoder_size']
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

        print "We have {} hidden units".format(num_units)

        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                            input_var=input_var)

        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units/4, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)

        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units/2, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)

        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)


        network = lasagne.layers.FlattenLayer(network)
        network = lasagne.layers.DenseLayer(network, 1,
                                             nonlinearity=lasagne.nonlinearities.sigmoid)

        return network

    