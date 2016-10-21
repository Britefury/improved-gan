import argparse
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import plotting
import cifar10_data
import improved_gan
from britefury_lasagne.dataset import balanced_subset_indices

# settings
SEED = 1
BATCH_SIZE = 100
UNLABELED_WEIGHT = 1.0
LEARNING_RATE = 0.0003
DATA_DIR = './data/cifar10'
N_LABELED = 4000 # Original code had COUNT=400, but thats *per class*, so 4000 total

# fixed random seeds
rng = np.random.RandomState(SEED)
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

print('Loading data...')
# load CIFAR-10
trainx_all, trainy_all = cifar10_data.load(DATA_DIR, subset='train')
testx, testy = cifar10_data.load(DATA_DIR, subset='test')

# select labeled data
train_indices = balanced_subset_indices(trainy_all, n_classes=10, n_samples=N_LABELED, shuffle=True, rng=rng)
trainx = trainx_all[train_indices]
trainy = trainy_all[train_indices]



class CIFAR10ImprovedGANSemiSupervisedClassifier (improved_gan.ImprovedGANSemiSupervisedClassifier):
    def generative_model(self, noise_size, noise_expr):
        # specify generative model
        input_layer = ll.InputLayer(shape=(None, noise_size), input_var=noise_expr)
        layers = [input_layer]
        layers.append(nn.batch_norm(ll.DenseLayer(layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
        layers.append(ll.ReshapeLayer(layers[-1], ([0],512,4,4)))
        layers.append(nn.batch_norm(nn.Deconv2DLayer(layers[-1], (None,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
        layers.append(nn.batch_norm(nn.Deconv2DLayer(layers[-1], (None,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
        layers.append(nn.weight_norm(nn.Deconv2DLayer(layers[-1], (None,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
        return improved_gan.GenerativeModel(layers=layers, input_layer=input_layer, final_layer=layers[-1])


    def discriminative_model(self):
        # specify discriminative model
        input_layer = ll.InputLayer(shape=(None, 3, 32, 32))
        layers = [input_layer]
        layers.append(ll.DropoutLayer(layers[-1], p=0.2))
        layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
        layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(layers[-1], 96, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
        layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(layers[-1], 96, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
        layers.append(ll.DropoutLayer(layers[-1], p=0.5))
        layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
        layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(layers[-1], 192, (3,3), pad=1, W=Normal(0.05), nonlinearity=nn.lrelu)))
        layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(layers[-1], 192, (3,3), pad=1, stride=2, W=Normal(0.05), nonlinearity=nn.lrelu)))
        layers.append(ll.DropoutLayer(layers[-1], p=0.5))
        layers.append(nn.weight_norm(dnn.Conv2DDNNLayer(layers[-1], 192, (3,3), pad=0, W=Normal(0.05), nonlinearity=nn.lrelu)))
        layers.append(nn.weight_norm(ll.NINLayer(layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
        layers.append(nn.weight_norm(ll.NINLayer(layers[-1], num_units=192, W=Normal(0.05), nonlinearity=nn.lrelu)))
        feature_layer = ll.GlobalPoolLayer(layers[-1])
        layers.append(feature_layer)
        return improved_gan.DiscriminativeModel(layers=layers, input_layer=input_layer, final_layer=layers[-1], feature_layer=feature_layer)


    def on_batch_complete(self, epoch):
        # generate samples from the model
        z = self.rng.uniform(size=(100, self.noise_size)).astype(np.float32)
        sample_x = self.samplefun(z)
        img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
        img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
        plotting.plot_img(img_tile, title='CIFAR10 samples', figsize=(8,8))
        plotting.plt.savefig("cifar_sample_feature_match.png")


print('Building...')
model = CIFAR10ImprovedGANSemiSupervisedClassifier(
    improved_gan.FeatureMatchingStabilizer(),
    n_classes=10, noise_size=100, unlabeled_weight=UNLABELED_WEIGHT, rng=rng)

print('Initialising...')
model.initialise(trainx_all)

print('Training...')
learning_rate_fn = lambda epoch: LEARNING_RATE * np.minimum(3. - epoch/400., 1.)
model.train(trainx, trainy, trainx_all, None, [testx, testy], num_epochs=1200,
            batch_size=128, learning_rate=learning_rate_fn)