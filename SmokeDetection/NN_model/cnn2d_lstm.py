import logging
import tensorflow as tf
from easydict import EasyDict
from tensorflow.contrib import rnn
from NN_model.libs import Model_skeleton, log_tensor_info
from smoke_detection_core.motion_detection import motion_detector_factory

def Cnn2d_lstm_conf():
    hparams = EasyDict()

    # Key parameters.
    # Data preprocessing
    hparams.is_standardization = True
    # Classes_num
    hparams.num_classes = 2
    # Epoches for train
    hparams.epoches = 5
    # Batch size
    hparams.batch_size = 32
    # Learning rate
    hparams.learning_rate = 0.001
    # Motion detector.
    hparams.motion_detector = 'background_substraction'

    # The method for adjusting learning rate, which is in ['constant', 'exponential']
    hparams.lr_mode = 'exponential_decay'
    # For exponential decay
    hparams.lr_decay_rate = 0.8
    hparams.lr_decay_steps = 200

    # The following hparams are for optimizer
    # The optimizer for the net, which is in ['sgd', 'mome', 'adam', 'rmsp']
    hparams.optimizer = 'adam'
    # For 'mome'
    hparams.mome_momentum = 0.8
    # For 'adam'
    hparams.adam_beta1 = 0.9
    hparams.adam_beta2= 0.95
    # For 'rmsp'
    hparams.rmsp_mometum = 0.8
    hparams.rmsp_decay = 0.9

    # Weight decay for L1 or L2 or L1_L2 normalization
    # The regularization mode is option in [None, 'L1', 'L2', 'L1_L2']
    hparams.regularization_mode = 'L2'
    # For 'L1' or 'L1_L2'
    hparams.L1_scale = 0.0005
    # For 'L2' or 'L1_L2'
    hparams.L2_scale = 0.0005

    # Keep prob for dropout
    hparams.keep_prob = 0.8

    # Epsilon for stability
    hparams.epsilon = 1e-8

    # Gradients will be normalizing to max 10.0. Larger than this value will be clipped.
    hparams.max_grad_norm = 1.0

    return hparams


class Cnn2d_lstm(Model_skeleton):
    def __init__(self, hparams):
        super(Cnn2d_lstm, self).__init__(hparams=hparams)
        self.motion_detector = motion_detector_factory().get_motion_detector(hparams.motion_detector)
        data_shape = [None, self.hparams.sample_sum_frames, self.hparams.block_size, self.hparams.block_size, 3]
        self.ph_data = tf.placeholder(dtype=tf.float32, shape=data_shape,name='ph_data')
        self.ph_label = tf.placeholder(dtype=tf.float32, shape=[None, self.hparams.num_classes], name='ph_label')
        self.ph_is_training = tf.placeholder(dtype=tf.bool, name='ph_is_training')
        log_tensor_info(self.ph_data)
        log_tensor_info(self.ph_label)
        log_tensor_info(self.ph_is_training)
        logging.info('Model initialization completed!')

        self._add_forward_graph()
        self._add_argmax_output_graph()
        self._add_loss_graph()
        self._add_train_graph()
        self._viz_key_data()
        self._count_trainable_parameters()

    # def _add_forward_graph(self):
    #     block_1 = tf.map_fn(fn=lambda input: self._conv2d_layer('block_1', input, 16, 5, 1), elems=self.ph_data, dtype=tf.float32)
    #     block_1 = self._bn_layer('bn_1', block_1)
    #     maxpool_1 = tf.map_fn(fn=lambda input: self._maxpool_layer('maxpool_1', input, 2, 2), elems=block_1, dtype=tf.float32)
    #     block_2 = tf.map_fn(fn=lambda input: self._conv2d_layer('block_2', input, 32, 3, 1), elems=maxpool_1, dtype=tf.float32)
    #     block_2 = self._bn_layer('bn_2', block_2)
    #     maxpool_2 = tf.map_fn(fn=lambda input: self._maxpool_layer('maxpool_2', input, 2, 2), elems=block_2, dtype=tf.float32)
    #     # block_3 = tf.map_fn(fn = lambda input: self._conv2d_layer('block_3', input, 64, 3, 1), elems=maxpool_2, dtype=tf.float32)
    #     # block_3 = self._bn_layer('bn_3', block_3)
    #     # maxpool_3 = tf.map_fn(fn=lambda input: self._maxpool_layer('maxpool_3', input, 2, 2), elems=block_3, dtype=tf.float32)
    #     block_4 = tf.map_fn(fn=lambda  input: self._conv2d_layer('block_4', input, 96, 3, 1), elems=maxpool_2, dtype=tf.float32)
    #     block_4 = self._bn_layer('bn_4', block_4)
    #     maxpool_4_stride = int(block_4.get_shape()[2])
    #     maxpool_4 = tf.map_fn(fn=lambda input: self._maxpool_layer('maxpool_4', input, maxpool_4_stride, maxpool_4_stride), elems=block_4, dtype=tf.float32)
    #
    #     lstm_in = tf.squeeze(maxpool_4, axis=[2, 3],name='lstm_in')
    #     lstm_in = tf.transpose(lstm_in, [1, 0, 2])
    #     lstm_in = tf.reshape(lstm_in, [-1, 96])
    #     lstm_in = tf.layers.dense(lstm_in, units=256, activation=None)
    #     lstm_in = tf.split(lstm_in, self.hparams.sample_sum_frames, 0)
    #     lstm_cell = rnn.BasicLSTMCell(num_units=256, forget_bias=1.0, state_is_tuple=True)
    #     drop_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.8)
    #     mlstm_cell = rnn.MultiRNNCell([drop_cell]*2, state_is_tuple=True)
    #     # init_state = mlstm_cell.zero_state(self.hparams.batch_size, dtype=tf.float32)
    #     outputs, final_state = rnn.static_rnn(mlstm_cell, lstm_in, dtype=tf.float32)
    #     h_state = outputs[-1]
    #     self.nn_output = self._fc_layer('nn_output', h_state, hiddens=self.hparams.num_classes)

    def _add_forward_graph(self):
        block_1 = tf.map_fn(fn=lambda input: self._conv2d_layer('block_1', input, 16, 5, 1), elems=self.ph_data, dtype=tf.float32)
        block_1 = self._bn_layer('bn_1', block_1)
        avgpool_1 = tf.map_fn(fn=lambda input: self._maxpool_layer('avgpool_1', input, 2, 2), elems=block_1, dtype=tf.float32)
        block_2 = tf.map_fn(fn=lambda input: self._conv2d_layer('block_2', input, 32, 3, 1), elems=avgpool_1, dtype=tf.float32)
        block_2 = self._bn_layer('bn_2', block_2)
        avgpool_2 = tf.map_fn(fn=lambda input: self._maxpool_layer('avgpool_2', input, 2, 2), elems=block_2, dtype=tf.float32)
        # block_3 = tf.map_fn(fn = lambda input: self._conv2d_layer('block_3', input, 64, 3, 1), elems=maxpool_2, dtype=tf.float32)
        # block_3 = self._bn_layer('bn_3', block_3)
        # maxpool_3 = tf.map_fn(fn=lambda input: self._maxpool_layer('maxpool_3', input, 2, 2), elems=block_3, dtype=tf.float32)
        block_4 = tf.map_fn(fn=lambda  input: self._conv2d_layer('block_4', input, 96, 3, 1), elems=avgpool_2, dtype=tf.float32)
        block_4 = self._bn_layer('bn_4', block_4)
        maxpool_4_stride = int(block_4.get_shape()[2])
        avgpool_4 = tf.map_fn(fn=lambda input: self._maxpool_layer('avgpool_4', input, maxpool_4_stride, maxpool_4_stride), elems=block_4, dtype=tf.float32)

        lstm_in = tf.squeeze(avgpool_4, axis=[2, 3],name='lstm_in')
        lstm_in = tf.transpose(lstm_in, [1, 0, 2])
        lstm_in = tf.reshape(lstm_in, [-1, 96])
        lstm_in = tf.layers.dense(lstm_in, units=256, activation=None)
        lstm_in = tf.split(lstm_in, self.hparams.sample_sum_frames, 0)
        lstm_cell = rnn.BasicLSTMCell(num_units=256, forget_bias=1.0, state_is_tuple=True)
        drop_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.8)
        mlstm_cell = rnn.MultiRNNCell([drop_cell]*2, state_is_tuple=True)
        # init_state = mlstm_cell.zero_state(self.hparams.batch_size, dtype=tf.float32)
        outputs, final_state = rnn.static_rnn(mlstm_cell, lstm_in, dtype=tf.float32)
        h_state = outputs[-1]
        self.nn_output = self._fc_layer('nn_output', h_state, hiddens=self.hparams.num_classes)


    def _conv2d_block(self, block_name, input_data, out_channels, kernel_size, stride, padding='SAME'):
        with tf.variable_scope(block_name):
            conv2d_out = self._conv2d_layer(block_name+'_conv2d', input_data, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=stride, padding='SAME')
            # bn_out = self._bn_layer(block_name+'_bn', conv2d_out)
            relu_out = self._relu_layer(block_name+'_relu', conv2d_out)
            return relu_out

