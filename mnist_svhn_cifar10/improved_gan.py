from collections import namedtuple
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
from britefury_lasagne import data_source


GenerativeModel = namedtuple('GenerativeModel', ['layers', 'input_layer', 'final_layer'])
DiscriminativeModel = namedtuple('DiscriminativeModel', ['layers', 'input_layer', 'final_layer', 'feature_layer'])


class AbstractStabilizer (object):
    N_GEN_TRAIN_REPEATS = 1
    REQUIRE_SEPARATE_UNLABELED_SAMPLES_FOR_GEN = False

    def additional_disc_layers(self, disc_model):
        return []

    def generator_update_function(self, disc_model, gen_model, batch_size_var, x_unlabeled, gen_output_train, loss_fake, lr):
        raise NotImplementedError('Abstract for type {}'.format(type(self)))


class FeatureMatchingStabilizer (AbstractStabilizer):
    REQUIRE_SEPARATE_UNLABELED_SAMPLES_FOR_GEN = True

    def generator_update_function(self, disc_model, gen_model, batch_size_var, x_unlabeled, gen_output_train, loss_fake, lr):
        output_unl = ll.get_output(disc_model.feature_layer, x_unlabeled, deterministic=False)
        output_gen = ll.get_output(disc_model.feature_layer, gen_output_train, deterministic=False)
        m1 = T.mean(output_unl,axis=0)
        m2 = T.mean(output_gen,axis=0)
        loss_gen = T.mean(abs(m1-m2)) # feature matching loss
        gen_params = ll.get_all_params(gen_model.layers, trainable=True)
        gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
        return th.function(inputs=[batch_size_var, x_unlabeled, lr], outputs=None, updates=gen_param_updates)


class MinibatchDiscriminationStabilizer (AbstractStabilizer):
    N_GEN_TRAIN_REPEATS = 3

    def __init__(self, minibatch_discrim_num_kernels=100):
        self.minibatch_discrim_num_kernels = minibatch_discrim_num_kernels

    def additional_disc_layers(self, disc_model):
        mb = nn.MinibatchLayer(disc_model.final_layer, num_kernels=self.minibatch_discrim_num_kernels)
        return [mb]

    def generator_update_function(self, disc_model, gen_model, batch_size_var, x_unlabeled, gen_output_train, loss_fake, lr):
        loss_gen = -T.mean(T.nnet.softplus(loss_fake))
        gen_params = ll.get_all_params(gen_model.final_layer, trainable=True)
        gen_param_updates = nn.adam_updates(gen_params, loss_gen, lr=lr, mom1=0.5)
        return th.function(inputs=[batch_size_var, lr], outputs=None, updates=gen_param_updates)


class ImprovedGANSemiSupervisedClassifier (object):
    def __init__(self, stabilizer, n_classes, noise_size, unlabeled_weight=1.0, rng=None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random
        self.stabilizer = stabilizer
        self.n_classes = n_classes
        self.noise_size = noise_size
        self.unlabeled_weight = unlabeled_weight
        self.batch_size_var = T.iscalar('batch_size')
        self.z_rng = MRG_RandomStreams(self.rng.randint(2 ** 15))

        self._build()

    def generative_model(self, noise_size, noise_expr):
        raise NotImplementedError('Abstract for {}'.format(type(self)))


    def discriminative_model(self):
        raise NotImplementedError('Abstract for {}'.format(type(self)))


    def _build_discriminator(self):
        disc_model = self.discriminative_model()
        if not isinstance(disc_model, DiscriminativeModel):
            raise TypeError('discriminative_model() should return a DiscriminativeModel instance')
        layers = disc_model.layers
        # Add extra layers required by stabilizer
        stabilizer_layers = self.stabilizer.additional_disc_layers(disc_model)
        layers = layers + stabilizer_layers
        # Add final classification layer
        layers.append(nn.weight_norm(ll.DenseLayer(disc_model.final_layer, num_units=self.n_classes, W=Normal(0.05), nonlinearity=None), train_g=True, init_stdv=0.1))
        return DiscriminativeModel(layers=layers, input_layer=disc_model.input_layer, final_layer=layers[-1],
                                   feature_layer=disc_model.feature_layer)


    def _build(self):
        noise_dim = (self.batch_size_var, self.noise_size)
        noise_expr = self.z_rng.uniform(size=noise_dim)
        gen_model = self.generative_model(self.noise_size, noise_expr)

        if not isinstance(gen_model, GenerativeModel):
            raise TypeError('generative_model() should return a GenerativeModel instance')

        disc_model = self._build_discriminator()

        # Variables
        labels = T.ivector()
        x_lab = T.tensor4()
        x_unl = T.tensor4()
        z_noise = T.matrix()
        lr = T.scalar()

        # costs
        temp = ll.get_output(gen_model.final_layer, deterministic=False, init=True)
        temp = ll.get_output(disc_model.layers[-1], x_lab, deterministic=False, init=True)
        init_updates = [u for l in gen_model.layers+disc_model.layers for u in getattr(l,'init_updates',[])]

        gen_output_train = ll.get_output(gen_model.final_layer)
        gen_output_generate = ll.get_output(gen_model.final_layer, inputs={gen_model.input_layer: z_noise})

        output_before_softmax_lab = ll.get_output(disc_model.layers[-1], x_lab, deterministic=False)
        output_before_softmax_unl = ll.get_output(disc_model.layers[-1], x_unl, deterministic=False)
        output_before_softmax_fake = ll.get_output(disc_model.layers[-1], gen_output_train, deterministic=False)

        loss_lab = output_before_softmax_lab[T.arange(self.batch_size_var),labels]
        loss_unlab = nn.log_sum_exp(output_before_softmax_unl)
        loss_fake = nn.log_sum_exp(output_before_softmax_fake)
        disc_loss_lab = -T.mean(loss_lab) + T.mean(T.mean(nn.log_sum_exp(output_before_softmax_lab)))
        disc_loss_unlab = -0.5*T.mean(loss_unlab) + 0.5*T.mean(T.nnet.softplus(loss_unlab)) + 0.5*T.mean(T.nnet.softplus(loss_fake))

        train_err = T.mean(T.neq(T.argmax(output_before_softmax_lab,axis=1),labels))

        # test error
        output_before_softmax = ll.get_output(disc_model.layers[-1], x_lab, deterministic=True)
        test_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),labels))

        # Theano functions for training the disc net
        disc_params = ll.get_all_params(disc_model.layers, trainable=True)
        disc_param_updates = nn.adam_updates(disc_params, disc_loss_lab + self.unlabeled_weight*disc_loss_unlab, lr=lr, mom1=0.5)
        disc_param_avg = [th.shared(np.cast[th.config.floatX](0.*p.get_value())) for p in disc_params]
        disc_avg_updates = [(a,a+0.0001*(p-a)) for p,a in zip(disc_params,disc_param_avg)]
        disc_avg_givens = [(p,a) for p,a in zip(disc_params,disc_param_avg)]
        self.init_param = th.function(inputs=[self.batch_size_var, x_lab], outputs=None, updates=init_updates) # data based initialization
        self.train_batch_disc = th.function(inputs=[self.batch_size_var,x_lab,labels,x_unl,lr], outputs=[disc_loss_lab, disc_loss_unlab, train_err], updates=disc_param_updates+disc_avg_updates)
        self.test_batch = th.function(inputs=[x_lab,labels], outputs=test_err, givens=disc_avg_givens)
        self.samplefun = th.function(inputs=[z_noise],outputs=gen_output_generate)

        # Theano functions for training the gen net
        self.train_batch_gen = self.stabilizer.generator_update_function(disc_model, gen_model, self.batch_size_var,
                                                                         x_unl, gen_output_train, loss_fake, lr)


    def initialise(self, train_X, N_init=500):
        (batch_X,) = data_source.batch_iterator([train_X], batchsize=N_init).next()
        self.init_param(N_init, batch_X)

    def _train_batch_iter(self, train_X, train_y, unlabeled_X, batch_size):
        # Initialise labeled indices array to empty; will fill later on
        lab_ndx = np.arange(0)
        lab_ndx_pos = 0

        N_unlab = len(unlabeled_X)
        N_train = len(train_X)

        # Indices for unlabeled samples
        unlab_disc_ndx = np.arange(N_unlab)
        self.rng.shuffle(unlab_disc_ndx)
        # Separate indices for generator if necessary
        if self.stabilizer.REQUIRE_SEPARATE_UNLABELED_SAMPLES_FOR_GEN:
            unlab_gen_ndx = np.arange(N_unlab)
            self.rng.shuffle(unlab_gen_ndx)

        for i in range(0, N_unlab, batch_size):
            # Select batch of unlabeled samples
            batch_unlab_disc_ndx = unlab_disc_ndx[i:i+batch_size]
            # Select separate batch for generator if necessary
            if self.stabilizer.REQUIRE_SEPARATE_UNLABELED_SAMPLES_FOR_GEN:
                batch_unlab_gen_ndx = unlab_gen_ndx[i:i+batch_size]
            # Get the size of the unlabeled batch; may not be equal to batch_size if we ran out of samples
            unlab_batch_size = batch_unlab_disc_ndx.shape[0]

            # Select batch of labeled samples and advance the position marker
            batch_lab_ndx = lab_ndx[lab_ndx_pos:lab_ndx_pos+unlab_batch_size]
            lab_ndx_pos += unlab_batch_size

            # If the batch of labeled samples is short it means we've run out of samples in the labeled set
            # so start over
            if batch_lab_ndx.shape[0] < unlab_batch_size:
                # Re-fill the labeled samples array and shuffle
                lab_ndx = np.arange(N_train)
                self.rng.shuffle(lab_ndx)
                # Set the position marker the position required to fill the labeled batch
                lab_ndx_pos = unlab_batch_size - batch_lab_ndx.shape[0]
                batch_lab_ndx = np.append(batch_lab_ndx, lab_ndx[:lab_ndx_pos], axis=0)

            batch_lab_X = train_X[batch_lab_ndx]
            batch_lab_y = train_y[batch_lab_ndx]
            batch_unlab_disc_X = unlabeled_X[batch_unlab_disc_ndx]
            if self.stabilizer.REQUIRE_SEPARATE_UNLABELED_SAMPLES_FOR_GEN:
                batch_unlab_gen_X = unlabeled_X[batch_unlab_gen_ndx]
            else:
                batch_unlab_gen_X = None

            assert batch_lab_X.shape[0] == batch_lab_y.shape[0]
            assert batch_lab_X.shape[0] == batch_unlab_disc_X.shape[0]
            if self.stabilizer.REQUIRE_SEPARATE_UNLABELED_SAMPLES_FOR_GEN:
                assert batch_lab_X.shape[0] == batch_unlab_gen_X.shape[0]


            yield (batch_lab_X, batch_lab_y, batch_unlab_disc_X, batch_unlab_gen_X)


    def train(self, train_X, train_y, unlabeled_X, val, test, num_epochs=100, batch_size=128, learning_rate=0.0003):
        for epoch in range(num_epochs):
            begin = time.time()
            if callable(learning_rate):
                lr = learning_rate(epoch)
            else:
                lr = learning_rate
            lr = np.cast[th.config.floatX](lr)


            # train
            loss_lab = 0.
            loss_unl = 0.
            train_err = 0.
            nr_batches_train = 1

            for batch_i, (batch_lab_X, batch_lab_y, batch_unlab_disc_X, batch_unlab_gen_X) in enumerate(
                    self._train_batch_iter(train_X, train_y, unlabeled_X, batch_size)):
                batch_N = batch_lab_X.shape[0]
                ll, lu, te = self.train_batch_disc(batch_N, batch_lab_X, batch_lab_y, batch_unlab_disc_X, lr)
                loss_lab += ll
                loss_unl += lu
                train_err += te

                for i in range(self.stabilizer.N_GEN_TRAIN_REPEATS):
                    if self.stabilizer.REQUIRE_SEPARATE_UNLABELED_SAMPLES_FOR_GEN:
                        self.train_batch_gen(batch_N, batch_unlab_gen_X, lr)
                    else:
                        self.train_batch_gen(batch_N, lr)

                nr_batches_train = batch_i + 1


            loss_lab /= nr_batches_train
            loss_unl /= nr_batches_train
            train_err /= nr_batches_train

            # val
            if val is not None:
                val_err = 0.
                nr_batches_val = 1
                for batch_i, (batch_x, batch_y) in enumerate(data_source.batch_iterator(val, batchsize=500)):
                    val_err += self.test_batch(batch_x, batch_y)
                    nr_batches_val = batch_i + 1
                val_err /= nr_batches_val
                val_err_string = ', val err = {:.4f}'.format(val_err)
            else:
                val_err_string = ''

            # test
            if test is not None:
                test_err = 0.
                nr_batches_test = 1
                for batch_i, (batch_x, batch_y) in enumerate(data_source.batch_iterator(test, batchsize=500)):
                    test_err += self.test_batch(batch_x, batch_y)
                    nr_batches_test = batch_i + 1
                test_err /= nr_batches_test
                test_err_string = ', test err = {:.4f}'.format(test_err)
            else:
                test_err_string = ''

            # report
            print("Iteration {}, time = {:.2f}s, loss_lab = {:.4f}, loss_unl = {:.2f}, train err = {:.4f}{}{}".format(
                epoch, time.time()-begin, loss_lab, loss_unl, train_err, val_err_string, test_err_string))
            sys.stdout.flush()

            # generate samples from the model
            self.on_batch_complete(epoch)


    def on_batch_complete(self, epoch):
        pass