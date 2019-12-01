from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import os, sys
from argparse import ArgumentParser

# Debug module
# from tensorflow.python import debug as tf_debug

import numpy as np
import warnings
from keras.datasets import mnist
from tensorflow.python.summary.writer.writer import FileWriter
import matplotlib.pyplot as plt

warnings.simplefilter('error', UserWarning)


class IWAE:
    def __init__(self, input_shape, batch_size, layer_specs, k_samples, lr, sess, small):
        self.data_ph = tf.placeholder(dtype=tf.float32, shape=(None, k_samples, input_shape))
        self.train_ph = tf.placeholder(dtype=tf.bool)
        self.tot_obj_loss = tf.placeholder(dtype=tf.float32)
        self.log2pi = tf.log(2 * np.pi)
        self.q_probs = []
        self.h_units = layer_specs
        self.batch_size = batch_size
        self.small = small
        self.init = tf.placeholder(dtype=tf.bool)
        self.k = k
        self.log_w = tf.zeros(dtype=tf.float32, shape=[batch_size, self.k])
        self.norm_w = tf.zeros_like(self.log_w)
        self.sess = sess
        self.recon = self.model(self.data_ph)
        self.loss, self.obj_loss = self.objective_function()
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999).minimize(self.obj_loss)

        self.summary = tf.Summary()
        loss_summary = tf.summary.scalar('Objective loss', self.tot_obj_loss)
        self.merge_op = tf.summary.merge_all()
        print('Logging to:', './logs/' + str(datetime.datetime.now()))
        self.writer = tf.summary.FileWriter('./logs/' + str(datetime.datetime.now()))

    def dense(self, x_, num_units, init_scale=0.01, scope_name=''):
        """
        Dense layer including Weight normalization and initialization
        as presented by (Kingma & Salimans, Weight normalization, 2016)
        based on code from: https://github.com/openai/weightnorm/blob/master/tensorflow/nn.py
        currently not giving any good desirable results
        :param x: input data
        :param num_units: number of units in the dense layer
        :param init_scale: initialization scale
        :param scope_name: name of current scope
        :return: data run through dense layer
        """
        with tf.variable_scope(scope_name):
            ema = tf.train.ExponentialMovingAverage(decay=0.998)
            if self.init is not False:
                V = tf.get_variable('V', shape=[int(x_.get_shape()[-1]), num_units], dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(0, 0.05), trainable=True)

                g = tf.get_variable('g', shape=[num_units], dtype=tf.float32,
                                      initializer=tf.constant_initializer(1.), trainable=True)

                b = tf.get_variable('b', shape=[num_units], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.), trainable=True)
            else:
                V = tf.get_variable('V')
                g = tf.get_variable('g')
                b = tf.get_variable('b')
                tf.assert_variables_initialized([V, g, b])

            ema.apply([V, g, b])
            g_ = tf.expand_dims(g, 0)
            g_ = tf.tile(g_, [self.k, 1])
            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(x_, V)
            scaler = g_ / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            b_ = tf.expand_dims(b, 0)
            b_ = tf.tile(b_, [self.k, 1])
            x = tf.reshape(scaler, [1, self.k, num_units]) * x + tf.reshape(b_, [1,  self.k, num_units])

            if self.init is not False:  # normalize x
                m_init, v_init = tf.nn.moments(x, [0])
                m_init = m_init[0]
                scale_init = init_scale / tf.sqrt(v_init + 1e-10)
                scale_init = scale_init[0]

                with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                    # x = tf.identity(x)
                    g_s = tf.expand_dims(g, 0)
                    g_s = tf.tile(g_s, [self.k, 1])
                    x = tf.matmul(x_, V)
                    scaler = g_s / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
                    b_ = tf.expand_dims(b, 0)
                    b_ = tf.tile(b_, [self.k, 1])
                    x = tf.reshape(scaler, [1, self.k, num_units]) * x + tf.reshape(b_, [1, self.k, num_units])
            return x

    def MLP_layer(self, x, mlp_units, out_dims, scope_name=''):
        """
        MLP layer with sampling built in
        :param x: input data
        :param mlp_units: dimensions of the MLP layers
        :param out_dims: output dimension for matching the next MLP layer
        :param scope_name: set the scope_name for WeightNorm, currently not working properly
        :return: nu, rho
        """
        # 2 regular linear dense layers with leaky Relu activations
        # x = self.dense(x, num_units=mlp_units, init_scale=1., scope_name=scope_name + '_dense1')
        x = tf.layers.dense(x, mlp_units)
        h_inter = tf.nn.leaky_relu(x, alpha=0.1)
        # h_i = self.dense(h_inter, num_units=mlp_units, init_scale=1., scope_name=scope_name + '_dense2')
        h_i = tf.layers.dense(h_inter, mlp_units)
        h_i = tf.nn.leaky_relu(h_i, alpha=0.1)

        # nu = self.dense(h_i, num_units=out_dims, init_scale=1., scope_name=scope_name + '_dense3')
        nu = tf.layers.dense(h_i, out_dims)
        # rho = 0.01 + tf.nn.softplus(self.dense(h_i, num_units=out_dims, init_scale=1., scope_name=scope_name + '_dense4'))
        rho = 0.01 + tf.nn.softplus(tf.layers.dense(h_i, out_dims))
        return nu, rho

    def sample_z(self, nu, rho, value=None, bern=False):
        """
        sample from N(nu, rho)
        :param nu: mean
        :param rho: stddev
        :param value: None or the latent variables from the corresponding encoder layer (if we are in the decoder layer)
        :param bern: Flag for using a bernoulli distribution
        :return: logprob(z|nu,rho) & z
        """
        # flag for using a bernoulli distribution
        if bern:
            sample_dist = tf.distributions.Bernoulli(logits=nu, dtype=tf.float32)
            nu_bern = sample_dist.mean()
            return nu_bern, self.bincrossentropy(value, nu)
        # reparametrization trick
        eps = tf.random_normal(tf.shape(nu), dtype=tf.float32)
        z_next = nu + rho*eps
        if value is not None:
            estimate = value
        else:
            estimate = z_next
        log2pi = 0.5*np.log(2*np.pi)
        logprob_z = (-tf.constant(log2pi, dtype=tf.float32))-\
                    0.5*(tf.reduce_sum(tf.square((estimate-nu)/rho) + 2*tf.log(rho), axis=-1))
        return z_next, logprob_z

    def bincrossentropy(self, x, x_hat):
        """
        calculate binary cross-entropy between true image and reconstruction
        :param x: true image
        :param x_hat: reconstructed image at the bernoulli layer of the decoder
        :return: binary cross-entropy
        """
        x_hat = tf.nn.sigmoid(x_hat)
        bce = x * tf.log(x_hat + 1e-8) + (1 - x) * tf.log(1 - x_hat + 1e-8)
        return tf.reduce_sum(bce, axis=-1)

    def calc_logw(self, q_logprob, p_logprob):
        """
        calculate the log weights
        :param q_logprob: output of a layer in q
        :param p_logprob: output of a layer in p
        :return: no return
        """
        self.log_w += p_logprob - q_logprob

    def calc_norm_tilde(self):
        """
        calculates the normalized importance weights
        :return: no return
        """
        log_w_max = tf.math.reduce_max(self.log_w, axis=-1, keepdims=True)
        log_w = tf.math.subtract(self.log_w, log_w_max)
        w = tf.math.exp(log_w)
        self.norm_w = tf.math.divide(w, tf.math.reduce_sum(w, axis=-1, keepdims=True))

    def objective_function(self):
        """
        Calculate the objective function loss
        :return: deprecated loss and objective function loss
        """
        k = tf.constant(self.k, dtype=tf.float32)
        with tf.name_scope('Loss'):
            # this loss is currently not used anywhere, deprecated
            self.calc_norm_tilde()
            loss = - tf.reduce_mean(tf.reduce_sum(self.norm_w * self.log_w, axis=-1))

            # objective loss over k-samples
            log_sum_w = tf.reduce_logsumexp(self.log_w, axis=-1)
            obj_loss = - tf.reduce_sum(tf.math.subtract(log_sum_w, tf.math.log(k)), axis=0)
        return loss, obj_loss

    def train(self, trn_data):
        trn_data = np.array([self.k * [x] for x in trn_data])
        _, recon, obj_loss, loss, log_w = self.sess.run([self.optimizer,
                                                                  self.recon,
                                                                  self.obj_loss,
                                                                  self.loss,
                                                                  self.log_w],
                                                        feed_dict={
                                                        self.train_ph: True,
                                                        self.data_ph: trn_data,
                                                        self.init: False
                                                    })

        return recon, obj_loss, loss, log_w

    def test(self, test_data):
        test_data = np.array([self.k * [x] for x in test_data])
        recon, obj_loss, loss, log_w = self.sess.run([self.recon,
                                                      self.obj_loss,
                                                      self.loss,
                                                      self.log_w],
                                                     feed_dict={
                                                         self.data_ph: test_data,
                                                         self.train_ph: False,
                                                         self.init: False
                                                     })
        return recon, obj_loss, loss

    def data_based_initialize(self, mb_data):
        test_data = np.array([self.k * [x] for x in mb_data])
        empt = self.sess.run([], feed_dict={self.data_ph: test_data, self.init: True})

    def model(self, q_z_next):
        """
        IWAE model structure for the Non-facturized case
        :param q_z_next: input data
        :return: returns a reconstructed image
        """
        self.log_w = tf.zeros_like(self.log_w)
        q_logprob_tot = 0
        p_logprob_tot = 0
        q_nu_next = None
        q_rho_next = None
        recon = None
        q_zs = [q_z_next]
        if self.small is True:
            mult = 2
        else:
            mult = 8
        # Encoder portion
        for mlp_units in self.h_units:
            with tf.name_scope('Q_MLP_layer'):
                q_dense_name = 'Q_MLP_layer_{}_'.format(mlp_units)
                q_nu_next, q_rho_next = self.MLP_layer(q_z_next, mlp_units=mult * mlp_units,
                                                       out_dims=4, scope_name=q_dense_name)
            with tf.name_scope('Q_stochastic_layer'):
                q_z_next, q_logprob = self.sample_z(q_nu_next, q_rho_next)
                q_logprob_tot += q_logprob
                q_zs.append(q_z_next)

        # account for prior ~ N(0,1)
        with tf.name_scope('Prior'):
            prior_nu = tf.zeros_like(q_nu_next)
            prior_rho = tf.ones_like(q_rho_next)
            _, prior_logprob = self.sample_z(prior_nu, prior_rho, q_z_next)
            p_logprob_tot += prior_logprob

        # Decoder portion
        for p_out, mlp_units, q_z_in, q_z_out in zip([784],
                                                     self.h_units[::-1],
                                                     q_zs[:0:-1],
                                                     q_zs[-2::-1]):
            # at last decoder layer, sample from Bernoulli dist
            if p_out == 784:
                bern = True
            else:
                bern = False
            with tf.name_scope('P_MLP_layer'):
                p_dense_name = 'P_MLP_layer_{}_'.format(mlp_units)
                p_nu, p_rho = self.MLP_layer(
                    q_z_in, mlp_units=2 * mlp_units, out_dims=p_out, scope_name=p_dense_name)
            with tf.name_scope('P_stochastic_layer'):
                p_z_next, p_logprob = self.sample_z(p_nu, p_rho, q_z_out, bern=bern)
                if bern:
                    recon = p_z_next
            p_logprob_tot += p_logprob

        with tf.name_scope('log_w'):
            self.calc_logw(q_logprob_tot, p_logprob_tot)
        return recon


def mb(x, batch_size):
    """
    Minibatch generator
    :param x: input data
    :param batch_size: desired batch size
    :return: yield a new batch each call
    """
    n_samples = x.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    while True:
        permutation = np.random.permutation(x.shape[0])
        for b in range(n_batches):
            batch_idx = permutation[b *
                                    batch_size:(b + 1) * batch_size]
            batch = x[batch_idx]
            if batch.shape[0] is not batch_size:
                continue
            yield batch


parser = ArgumentParser("Tensorflow implementation of IWAE in TMC-paper from NeurIPS 2019")
parser.add_argument('-k',           dest='k',          type=int, default=5,    help="Option for choosing k")
parser.add_argument('--epochs',     dest='epochs',     type=int, default=1200, help="Option for choosing number of epochs")
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,  help="Option for choosing batch size")
parser.add_argument('--model_type', dest='model_type', type=str, default='small', help="Option for using small or large model")
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-3, help="Option for setting learning rate")
args = parser.parse_args()

print("Batch size: ", args.batch_size)
print("Number of epochs: ", args.epochs)
print("Model type: ", args.model_type)
print("k: ", args.k)
print("Learning rate: ", args.learning_rate)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
batch_size = 128
# TODO TEST WITH k = 5, k = 20, k = 50, k = 100
model_type = args.model_type
if model_type == 'small':
    small = True
else:
    small = False
lr = args.learning_rate
batch_size = args.batch_size
k = args.k
epochs = args.epochs
save_path = 'Vanilla_IWAE_model_non_fac_{}_k_{}'.format(model_type, k)
if not os.path.exists(save_path):
    os.mkdir(save_path)
with tf.Session() as sess:
    IWAE_net = IWAE(batch_size=batch_size, input_shape=784, k_samples=k, layer_specs=[100],
                    lr=lr, sess=sess, small=small)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    x_gen = mb(x_train, batch_size)
    x_gen_test = mb(x_test, batch_size)
    # x_gen_init = mb(x_train, batch_size)
    test_err = []
    try:
        for epoch in range(1, epochs+1):
            #  used for the WeightNorm initialization, our implementation is flawed and not used
            # if epoch == 1:
            #     init_batch = next(x_gen_init).reshape(batch_size, 784)
            #     IWAE_net.data_based_initialize(init_batch)

            # iterate enough times to see all of the training data each epoch 1 -> (len(train_data)/batch_size)
            for mb_epoch in range(1, 470):
                x_batch = next(x_gen).reshape(batch_size, 784)
                recon, obj_loss, loss, log_w = IWAE_net.train(x_batch)
            test_batch_counter = 0
            batch_test_err = 0

            # iterate enough times to see all of the test data each epoch 1 -> (len(test_data)/batch_size)
            for test_epoch in range(1, 80):
                x_batch_test = next(x_gen_test).reshape(batch_size, 784)
                test_batch_counter += x_batch_test.shape[0]
                recon, obj_loss, loss = IWAE_net.test(x_batch_test)
                batch_test_err += obj_loss

            testing_err = batch_test_err/int(test_batch_counter)  # normalize total error over the nr of batch samples
            summary = IWAE_net.sess.run(IWAE_net.merge_op, feed_dict={IWAE_net.tot_obj_loss: testing_err})
            IWAE_net.writer.add_summary(summary, global_step=epoch)
            # ugly hack for resetting the loss between epochs, only needed for tensorboard
            summary = IWAE_net.sess.run(IWAE_net.merge_op, feed_dict={IWAE_net.tot_obj_loss: 0})
            test_err.append(testing_err)

            print('=====> Objective loss at epoch {}: {}'.format(str(epoch), str(testing_err)))
            if epoch == epochs:
                # save model at end of runs
                print('got to end for model IWAE non-factorized {} with k: {}'.format(model_type, k))
                total_obj_loss_model = np.array(test_err)
                np.save(save_path+"/tot_obj_loss_k_{}_non_fac_{}_vanilla_IWAE".format(k, model_type), total_obj_loss_model)
                saver.save(sess,
                           save_path+"/model_Vanilla_IWAE_forward_non_fac_{}_with_k{}.ckpt".format(model_type, k))
        print(test_err)

    except KeyboardInterrupt:
        # possibility to save model before all epochs have run
        print('Stopped training and testing at epoch {} for model IWAE non-factorized {} with k: {}'.format(epoch,
                                                                                                            model_type,
                                                                                                            k))
        total_obj_loss_model = np.array(test_err)
        np.save(save_path + "/tot_obj_loss_k_{}_non_fac_{}_vanilla_IWAE".format(k, model_type), total_obj_loss_model)
        saver.save(sess,
                   save_path + "/model_Vanilla_IWAE_forward_non_fac_{}_with_k{}.ckpt".format(model_type, k))