import tensorflow as tf

from .mann_cell import MANNCell



class MANN():
    def __init__(self, learning_rate = 1e-3, input_size = 20 * 20, memory_size = 128, memory_dim = 40,
                 controller_size = 200, nb_reads = 4, num_layers = 1, nb_classes = 5, nb_samples_per_class = 10, batch_size = 16, model="MANN"):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.controller_size = controller_size
        self.num_layers = num_layers
        self.nb_reads = nb_reads
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.batch_size = batch_size
        self.model = model

        self.image = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.nb_classes * self.nb_samples_per_class, self.input_size], name="input_var")
        self.label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.nb_classes * self.nb_samples_per_class], name="target_var")


    def build_model(self):
        input_var = self.image
        target_var = self.label

        one_hot_target = tf.one_hot(target_var, self.nb_classes, axis=-1)
        offset_target_var = tf.concat([tf.zeros_like(tf.expand_dims(one_hot_target[:, 0], 1)), one_hot_target[:,:-1]], axis=1)
        ntm_input = tf.concat([input_var, offset_target_var], axis=2)

        if self.model == "LSTM":
            def rnn_cell(rnn_size):
                return tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(self.controller_size) for _ in range(self.num_layers)])
            hidden_dim = self.controller_size
        elif self.model == "MANN":
            cell = MANNCell(lstm_size=self.controller_size, memory_size=self.memory_size, memory_dim=self.memory_dim, nb_reads=self.nb_reads)
            hidden_dim = self.controller_size + self.nb_reads * self.memory_dim

        state = cell.zero_state(self.batch_size, tf.float32)
        output, cell_state = tf.scan(lambda init, elem: cell(elem, init[1]), elems=tf.transpose(ntm_input, perm=[1, 0, 2]), initializer=(tf.zeros(shape=(self.batch_size, hidden_dim)), state))
        output = tf.transpose(output, perm=[1, 0, 2])

        with tf.variable_scope("o2o"):
           output = tf.layers.dense(
                inputs=output,
                units=self.nb_classes,
                kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
            )

        self.output = tf.nn.softmax(output, dim=2)
        self.output = tf.reshape(self.output, shape=(self.batch_size, self.nb_classes * self.nb_samples_per_class, self.nb_classes))
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_target, logits=output), axis=1))

        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize((self.loss))

