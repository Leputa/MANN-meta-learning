import tensorflow as tf
import numpy as np
from mann.utils.tf_utils import variable_one_hot


class MANNCell():
    def __init__(self, lstm_size, memory_size, memory_dim, nb_reads,
                 gamma=0.95, reuse=False):
        self.lstm_size = lstm_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.nb_reads = nb_reads
        self.reuse = reuse
        self.step = 0
        self.gamma = gamma
        self.controller = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)


    def __call__(self, input, prev_state):
        M_prev, r_prev, controller_state_prev, wu_prev, wr_prev = \
            prev_state["M"], prev_state["read_vector"], prev_state["controller_state"], prev_state["wu"], prev_state["wr"]

        controller_input = tf.concat([input, wr_prev], axis=-1)
        with tf.variable_scope("controller", reuse=self.reuse):
            controller_hidden_t, controller_state_t = self.controller(controller_input, controller_state_prev)

        parameter_dim_per_head = self.memory_dim * 2 + 1
        parameter_total_dim = parameter_dim_per_head * self.nb_reads  # []

        with tf.variable_scope("o2p", reuse=(self.step > 0) or self.reuse):
            parameter = tf.layers.dense(
                inputs=controller_hidden_t,
                units=parameter_total_dim,
                kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
            )

        indices_prev, wlu_prev = self.least_used(wu_prev)

        k = tf.tanh(parameter[:, 0:self.nb_reads * self.memory_dim], name="k")
        a = tf.tanh(parameter[:, self.nb_reads * self.memory_dim: 2 * self.nb_reads * self.memory_dim], name="a")
        sig_alpha = tf.sigmoid(parameter[:, -self.nb_reads: ], name="sig_alpha")

        wr_t = self.read_head_addressing(k, M_prev)
        ww_t = self.write_head_addressing(sig_alpha, wr_prev, wlu_prev)

        wu_t = self.gamma * wu_prev + tf.reduce_sum(wr_t, axis=1) + tf.reduce_sum(ww_t, axis=1)

        # "Prior to writing to memory, the least used memory location set to zero"
        M_t = M_prev * tf.expand_dims(1. - tf.one_hot(indices_prev[:, -1], self.memory_size), dim=2)
        M_t = M_t + tf.matmul(tf.transpose(ww_t, perm=[0,2,1]), tf.reshape(a, shape=(a.get_shape()[0], self.nb_reads, self.memory_dim)))

        r_t = tf.reshape(tf.matmul(wr_t, M_t), shape=(r_prev.get_shape()[0], self.nb_reads * self.memory_dim))


        state = {
            "M": M_t,
            "read_vector": r_t,
            "controller_state": controller_state_t,
            "wu": wu_t,
            "wr": tf.reshape(wr_t, shape=(wr_t.get_shape()[0], self.nb_reads * self.memory_size)),
        }

        NTM_output = tf.concat([controller_hidden_t, r_t], axis=-1)

        self.step += 1
        return NTM_output, state


    def read_head_addressing(self, k, M_prev, eps=1e-8):
        with tf.variable_scope("read_head_addressing"):
            k = tf.reshape(k, shape=(k.get_shape()[0], self.nb_reads, self.memory_dim))
            inner_product = tf.matmul(k, tf.transpose(M_prev, [0, 2, 1]))

            k_norm = tf.sqrt(tf.expand_dims(tf.reduce_sum(tf.square(k), 2), 2))
            M_norm = tf.sqrt(tf.expand_dims(tf.reduce_sum(tf.square(M_prev), 2), 1))

            norm_product = k_norm * M_norm
            K = inner_product / (norm_product + eps)
            return tf.nn.softmax(K)

    def write_head_addressing(self, sig_alpha, wr_prev, wlu_prev):
        with tf.variable_scope("write_head_addressing"):
            sig_alpha = tf.expand_dims(sig_alpha, axis=-1)
            wr_prev = tf.reshape(wr_prev, shape=(wr_prev.get_shape()[0], self.nb_reads, self.memory_size))
            return sig_alpha * wr_prev + (1. - sig_alpha) * tf.expand_dims(wlu_prev, axis=1)

    def least_used(self, w_u):
        _, indices = tf.nn.top_k(w_u, k=self.memory_size)
        wlu = tf.cast(tf.slice(indices, [0, self.memory_size - self.nb_reads], [w_u.get_shape()[0], self.nb_reads]), dtype=tf.int32)
        wlu = tf.reduce_sum(tf.one_hot(wlu, self.memory_size), axis=1)
        return indices, wlu

    def zero_state(self, batch_size, dtype):
        with tf.variable_scope("init", reuse=self.reuse):
            M_0 = tf.constant(np.ones([batch_size, self.memory_size, self.memory_dim]) * 1e-6, dtype=tf.float32)
            r_0 = tf.zeros(shape=(batch_size, self.nb_reads * self.memory_dim))
            controller_state_0 = self.controller.zero_state(batch_size, dtype)
            wu_0 = variable_one_hot(shape=(batch_size, self.memory_size))
            wr_0 = variable_one_hot(shape=(batch_size, self.memory_size * self.nb_reads))

        state ={
            "M": M_0,
            "read_vector":r_0,
            "controller_state": controller_state_0,
            "wu": wu_0,
            "wr": wr_0,
        }

        return state
