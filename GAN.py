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
        self.last_loss = [0,0]

    def initialise(self):
        #TODO: build generator, discriminator
        generator_fn, discriminator_fn = self.build_network()

        self.generator = generator_fn()
        self.discriminator = discriminator_fn()

    def reload(self, model_file):
        
        print "reloading..."

        self.options = pkl.load(open(model_file + '.pkl'))
        self.initialise()
        
        # Reloading the generator
        with np.load(model_file + "_generator.npy.npz") as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.generator, param_values)

        # Reloading th ediscriminator
        with np.load(model_file + "_discriminator.npy.npz") as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.discriminator, param_values)


    def save(self, model_file):
        np.savez(model_file+"_generator.npy", *lasagne.layers.get_all_param_values(self.generator))
        np.savez(model_file+"_discriminator.npy", *lasagne.layers.get_all_param_values(self.discriminator))
        option_file = model_file + '.pkl'
        pkl.dump(self.options, open(option_file, 'w'))

    def build_generator(self, input_var=None):
        num_units = self.options['generator_num_units']
        encoder_size = self.options['generator_encoder_size']
        noise_size = self.options['noise_size']

        print "We have {} hidden units".format(num_units)

        network = lasagne.layers.InputLayer(shape=(None, noise_size),
                                            input_var=input_var)

        network = lasagne.layers.ReshapeLayer(network, (-1, noise_size, 1, 1))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(4, 4),
                                                       stride=(1, 1)))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units/2,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2, output_size=8))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units/4,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2, output_size=16))

        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2,
                                                       nonlinearity=lasagne.nonlinearities.sigmoid, 
                                                       output_size=32)
        
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

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=num_units/2, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=num_units, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.FlattenLayer(network)
        network = lasagne.layers.DenseLayer(network, 1,
                                             nonlinearity=lasagne.nonlinearities.sigmoid)

        return network

    def build_network(self):
        return self.build_generator, self.build_discriminator

    def get_loss(self, disc_real, disc_sample, real_img, sample):

        #WGAN loss
        use_wgan = self.options['use_wgan']
        l2_penalty = self.options['l2_penalty']
        
        if use_wgan:
            print "Using the wgan loss"
            discriminator_loss=-0.5*((disc_real-disc_sample).mean())
            gan_generator_loss=-0.5*(disc_sample.mean())
        else:
            discriminator_loss = -(T.log(disc_real) + T.log(1.-disc_sample)).mean()
            gan_generator_loss = -T.log(disc_sample).mean()

        l2_loss = (l2_penalty)*lasagne.objectives.squared_error(sample, real_img).mean()
        generator_loss = gan_generator_loss + l2_loss
        return discriminator_loss, generator_loss, [gan_generator_loss, l2_loss]
    
    def _get_sample(self, input_vars):
        return lasagne.layers.get_output(self.generator, *input_vars)
    
    def _get_disc_score(self, sample, inputs_var):
        return lasagne.layers.get_output(self.discriminator, sample)

    def compile_theano_func(self, lr):
        # Prepare Theano variables for inputs and targets
        inputs, input_vars = self._get_inputs()
        origin_real_img = T.tensor4('real_img')

        real_img = origin_real_img.dimshuffle((0, 3, 1, 2))

        # Generator
        sample = self._get_sample(input_vars)

        #Discriminator
        disc_real = self._get_disc_score(real_img, input_vars)
        disc_sample = self._get_disc_score(sample, input_vars)

        # Our loss
        discriminator_loss, generator_loss, debug_values = self.get_loss(disc_real, disc_sample, real_img, sample)

        # generator update
        generator_params = lasagne.layers.get_all_params(self.generator, trainable=True)
        generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=lr, beta1=0.5)

        #Discriminator update
        discriminator_params = lasagne.layers.get_all_params(self.discriminator, trainable=True)
        discriminator_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=lr, beta1=0.5)


        print "Computing the functions..."


        self.train_generator_fn = None
        self.train_discriminator_fn = None
        self.train_generator_fn = theano.function(inputs + [origin_real_img], [generator_loss]+debug_values,
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

        noise_size = self.options['noise_size']
        noise = np.random.normal(size=(len(imgs), noise_size))

        use_wgan = self.options['use_wgan']
        
        [disc_loss] = self.train_discriminator_fn(noise, target)
        gen_loss = self.train_generator_fn(noise, target)

        if use_wgan:
            discriminator_params_values=lasagne.layers.get_all_param_values(self.discriminator, trainable=True)
            clamped_weights= [np.clip(w, -0.05, 0.05) for  w in discriminator_params_values]
            lasagne.layers.set_all_param_values(self.discriminator, clamped_weights, trainable=True)
        
        return disc_loss, gen_loss

    def get_generation_fn(self):
        
        noise_size = self.options['noise_size']
        
        def val_fn(imgs, target, caps):
            
            noise = np.random.uniform(size=(len(imgs), noise_size))
            res = self.generate_sample_fn(noise)
            return 0, res[0]

        return val_fn
    
class GAN64(GAN):
    
    def build_generator(self, input_var=None):
        num_units = self.options['generator_num_units']
        encoder_size = self.options['generator_encoder_size']
        noise_size = self.options['noise_size']

        print "We have {} hidden units".format(num_units)

        network = lasagne.layers.InputLayer(shape=(None, noise_size),
                                            input_var=input_var)

        network = lasagne.layers.ReshapeLayer(network, (-1, noise_size, 1, 1))
        network = lasagne.layers.BatchNormLayer(network)
        
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(4, 4),
                                                       stride=(1, 1))

        network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units/2,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2, output_size=8)

        network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units/4,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2, output_size=16)
                
        network = lasagne.layers.BatchNormLayer(network)
        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units/8,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2, output_size=32)

        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2,
                                                       nonlinearity=lasagne.nonlinearities.sigmoid, 
                                                       output_size=64)
        return network

    def build_discriminator(self, input_var=None):
        num_units = self.options['discriminator_num_units']
        encoder_size = self.options['discriminator_encoder_size']
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

        print "We have {} hidden units".format(num_units)

        network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                            input_var=input_var)
        
        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units/8, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)

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
    
    def train(self, imgs, target, caps):
        
        img_complete = utils.put_in_middle(imgs, target)

        noise_size = self.options['noise_size']
        noise = np.random.normal(size=(len(imgs), noise_size))

        use_wgan = self.options['use_wgan']
        
        [disc_loss] = self.train_discriminator_fn(noise, img_complete)
        gen_loss = self.train_generator_fn(noise, img_complete)

        if use_wgan:
            discriminator_params_values=lasagne.layers.get_all_param_values(self.discriminator, trainable=True)
            clamped_weights= [np.clip(w, -0.05, 0.05) for  w in discriminator_params_values]
            lasagne.layers.set_all_param_values(self.discriminator, clamped_weights, trainable=True)
        
        return gen_loss, disc_loss
    
class BICGAN(GAN):
        #Under cosntruction
    
    def build_generator(self, contour=None, noise=None):
        num_units = self.options['generator_num_units']
        encoder_size = self.options['generator_encoder_size']
        noise_size = self.options['noise_size']

        print "We have {} hidden units".format(num_units)
        
        #Our encoder
        self._contour_input = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                            input_var=contour)
        
        #encoder = lasagne.layers.BatchNormLayer(self._contour_input)
        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(self._contour_input, num_filters=num_units/8, filter_size=(5, 5),
                                             stride=2, pad=2))

        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder, num_filters=num_units/4, filter_size=(5, 5),
                                             stride=2, pad=2))

        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder, num_filters=num_units/2, filter_size=(5, 5),
                                             stride=2, pad=2))

        encoder = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(encoder, num_filters=num_units, filter_size=(5, 5),
                                             stride=2, pad=2))

        encoder = lasagne.layers.FlattenLayer(encoder)
        

        self._noise_input = lasagne.layers.InputLayer(shape=(None, noise_size),
                                            input_var=noise)
        
        # Merging the encoder and the noise.
        network = lasagne.layers.ConcatLayer([self._noise_input, encoder])

        network = lasagne.layers.ReshapeLayer(network, (-1, noise_size + num_units*4*4, 1, 1))

        network = lasagne.layers.batch_norm(
            lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(4, 4),
                                                 stride=(1, 1)))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units / 2,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=8))

        network = lasagne.layers.batch_norm(lasagne.layers.TransposedConv2DLayer(network, num_filters=num_units / 4,
                                                                                 filter_size=(5, 5), stride=(2, 2),
                                                                                 crop=2, output_size=16))

        network = lasagne.layers.TransposedConv2DLayer(network, num_filters=3,
                                                       filter_size=(5, 5), stride=(2, 2), crop=2,
                                                       nonlinearity=lasagne.nonlinearities.sigmoid,
                                                       output_size=32)


        return network

    def build_discriminator(self, input_var=None):
        num_units = self.options['discriminator_num_units']
        encoder_size = self.options['discriminator_encoder_size']
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

        print "We have {} hidden units".format(num_units)

        network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                            input_var=input_var)
        
        #We have one aditionnal layer.
        
        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units/16, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=num_units/8, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=num_units/4, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=num_units/2, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=num_units, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu))

        network = lasagne.layers.FlattenLayer(network)
        network = lasagne.layers.DenseLayer(network, 1,
                                             nonlinearity=lasagne.nonlinearities.sigmoid)

        return network
    
    def _get_sample(self, input_vars):
        return lasagne.layers.get_output(self.generator,
                                             {self._noise_input:input_vars[0], self._contour_input:input_vars[1]})
    
    def _get_disc_score(self, sample, inputs_var):
        
        # We set the middle
        contour = inputs_var[1] 
        center = (contour.shape[2] / 2, contour.shape[3] / 2)
        
        contour = T.set_subtensor(contour[:, :, center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16],  sample)
        
        return lasagne.layers.get_output(self.discriminator, contour)
    
    def _get_inputs(self):

        noise = T.matrix('noise')
        contour = T.tensor4('contour')
        
        contour_var = contour.transpose((0, 3, 1, 2))
        return [noise, contour], [noise, contour_var]
    
    def train(self, imgs, target, caps):
    
        noise_size = self.options['noise_size']
        noise = np.random.uniform(size=(len(imgs), noise_size))
        use_wgan = self.options['use_wgan']
        
        [disc_loss] = self.train_discriminator_fn(noise, imgs, target)
        self.last_loss[0] = disc_loss

        if False or self.schedule == 0:
            rval = self.train_generator_fn(noise, imgs, target)
            self.last_loss[1] = rval

        self.schedule = (self.schedule + 1) % 2
        
        if use_wgan:
            discriminator_params_values=lasagne.layers.get_all_param_values(self.discriminator, trainable=True)
            clamped_weights= [np.clip(w, -0.05, 0.05) for  w in discriminator_params_values]
            lasagne.layers.set_all_param_values(self.discriminator, clamped_weights, trainable=True)

        return self.last_loss
        #return [disc_loss, gen_loss]

    def get_generation_fn(self):
        
        noise_size = self.options['noise_size']
        
        def val_fn(imgs, target, caps):
            
            noise = np.random.uniform(size=(len(imgs), noise_size))
            res = self.generate_sample_fn(noise, imgs)

            return 0, res[0]

        return val_fn
    
class ConditionalGAN(BICGAN):

    def build_discriminator(self, input_var=None):
        num_units = self.options['discriminator_num_units']
        encoder_size = self.options['discriminator_encoder_size']
        lrelu = lasagne.nonlinearities.LeakyRectify(0.2)

        print "We have {} hidden units".format(num_units)

        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                            input_var=input_var)

        # We have one aditionnal layer.
        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units / 8, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)

        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units / 4, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)

        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units / 2, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)

        network = lasagne.layers.Conv2DLayer(network, num_filters=num_units, filter_size=(5, 5),
                                             stride=2, pad=2, nonlinearity=lrelu)
        network = lasagne.layers.BatchNormLayer(network)

        network = lasagne.layers.FlattenLayer(network)
        network = lasagne.layers.DenseLayer(network, 1,
                                            nonlinearity=lasagne.nonlinearities.sigmoid)

        return network

    def _get_disc_score(self, sample, inputs_var):
        return lasagne.layers.get_output(self.discriminator, sample)