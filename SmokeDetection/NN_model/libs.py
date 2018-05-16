import logging
from functools import reduce
import numpy as np
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool

def log_tensor_info(tensor):
    tensor_name = tensor.name
    tensor_shape = tensor.get_shape()
    info = tensor_name+str(tensor_shape)
    logging.info(info)

class Model_skeleton(object):
    def __init__(self, hparams):
        self.hparams = hparams

    def _add_forward_graph(self):
        logging.error('Model _add_forward_graph() is need to rewrite!')
        raise NotImplementedError

    def _add_argmax_output_graph(self):
        with tf.variable_scope('argmax_output'):
            softmax_output = tf.nn.softmax(self.nn_output, dim=1)
            self.argmax_output = tf.argmax(softmax_output, axis=1)
        logging.info('Add argmax output graph completed!')

    def _add_loss_graph(self):
        with tf.variable_scope('cross_entropy_loss'):
            batch_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.ph_label, logits=self.nn_output)
            self.classification_loss = tf.reduce_mean(batch_cross_entropy)
            tf.add_to_collection('losses_collection', self.classification_loss)
        if self.hparams.regularization_mode is not None:
            with tf.variable_scope('normalization_loss'):
                if self.hparams.regularization_mode == 'L2':
                    norm_func = tf.nn.l2_loss
                else:
                    logging.error('Current regularization_mode: {} is not supported, please check your configure!'.format(self.hparams.regularization_mode))

                for var in tf.trainable_variables():
                    var_regularization_loss = tf.multiply(norm_func(var), self.hparams.L2_scale)
                    tf.add_to_collection('losses_collection', var_regularization_loss)
        self.total_loss = tf.add_n(tf.get_collection('losses_collection'), name='total_loss')
        logging.info('Add loss graph completed!')

    def _add_train_graph(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

        # Set learning rate.
        if self.hparams.lr_mode == 'constant':
            self.learning_rate = self.hparams.learning_rate
        elif self.hparams.lr_mode == 'exponential_decay':
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.hparams.learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=self.hparams.lr_decay_steps,
                                                            decay_rate=self.hparams.lr_decay_rate)
        else:
            logging.error('Current lr_mode: {} is not supported, please check your configure!'.format(self.hparams.lr_mode))

        # Set optimizer.
        optimizer = None
        if self.hparams.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.hparams.optimizer =='mome':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.hparams.mome_momentum, use_nesterov=True)
        elif self.hparams.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.hparams.adam_beta1, beta2=self.hparams.adam_beta2)
        elif self.hparams.optimizer == 'rmsp':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.hparams.rmsp_decay, momentum=self.hparams.rmsp_momentum)
        else:
            logging.error('Current optimizer: {} is not supported, please check your configure!'.format(self.hparams.optimizer))

        # Train operation.
        self.train_op = optimizer.minimize(self.total_loss, var_list=tf.trainable_variables(), name='train_op', global_step=self.global_step)
        logging.info('Add train graph completed!')

    def _viz_key_data(self):
        tf.summary.scalar('classification_loss', self.classification_loss)
        tf.summary.scalar('total_loss',self.total_loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        logging.info('Add visualized key data completed!')

    def _count_trainable_parameters(self):
        total_params = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            var_params = reduce(lambda x, y: x * y, shape)
            total_params += int(var_params)
            shape_str = str(shape.as_list())
            logging.info(var.name + ' number:%d--' % var_params + shape_str)
        logging.info('Total trainable parameters are:%d' % total_params)

    ##########--Basic NN construction unit--start--##########
    def _variable(self, name, shape, initializer, trainable=True):
        var = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer, trainable=trainable)
        return var

    def _conv2d_layer(self, layer_name, input_data, out_channels, kernel_size, stride, add_bias=True, padding='SAME'):
        with tf.variable_scope(layer_name):
            kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
            in_channels = int(input_data.get_shape()[-1])
            kernel_shape = [kernel_size, kernel_size, in_channels, out_channels]
            kernel = self._variable(layer_name + '_kernel', shape=kernel_shape, initializer=kernel_init)
            strides = [1, stride, stride, 1]
            output = tf.nn.conv2d(input_data, filter=kernel, strides=strides, padding=padding, name=layer_name+'_output')
            if add_bias:
                dim = int(output.get_shape()[-1])
                bias = tf.get_variable(layer_name + '_bias', shape=[dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                output = tf.nn.bias_add(output, bias)
            log_tensor_info(output)
            return output

    def _maxpool_layer(self, layer_name, input_data, kernel_size, stride, padding='VALID'):
        with tf.variable_scope(layer_name):
            ksize = [1, kernel_size, kernel_size, 1]
            strides = [1, stride, stride, 1]
            output = tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding=padding, name=layer_name+'_output')
            log_tensor_info(output)
            return output

    def _avgpool_layer(self, layer_name, input_data, kernel_size, stride, padding='VALID'):
        with tf.variable_scope(layer_name):
            ksize = [1, kernel_size, kernel_size, 1]
            strides = [1, stride, stride, 1]
            output = tf.nn.avg_pool(input_data, ksize=ksize, strides=strides, padding=padding, name=layer_name+'_output')
            log_tensor_info(output)
            return output

    # def _conv3d_layer(self, layer_name, input_data, out_channels, kernel, stride, add_bias=True, padding='SAME'):
    #     with tf.variable_scope(layer_name):
    #         kernel_init = tf.contrib.layers.xavier_initializer()
    #         in_channels = int(input_data.get_shape()[-1])
    #         kernel_shape = kernel.extend([in_channels, out_channels])
    #         kernel = self._variable(layer_name + '_kernel', shape=kernel_shape, initializer=kernel_init)
    #         strides = [1, stride[0], stride[1], stride[2], 1]
    #         output = tf.nn.conv3d(input_data, filter=kernel, strides=strides, padding=padding, name=layer_name+'_output')
    #         if add_bias:
    #             dim = int(output.get_shape()[-1])
    #             bias = tf.get_variable(layer_name + '_bias', shape=[dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    #             output = tf.nn.bias_add(output, bias)
    #         log_tensor_info(output)
    #         return output
    def _conv3d_layer(self, layer_name, input_data, out_channels, kernel_size, stride, add_bias=False, padding='SAME'):
        with tf.variable_scope(layer_name):
            kernel_init = tf.contrib.layers.xavier_initializer()
            in_channels = int(input_data.get_shape()[-1])
            kernel_shape = [kernel_size[0], kernel_size[1], kernel_size[2], in_channels, out_channels]
            kernel = self._variable(layer_name + '_kernel', shape=kernel_shape, initializer=kernel_init)
            output = tf.nn.conv3d(input_data, filter=kernel, strides=[1, stride[0], stride[1], stride[2], 1],
                                  padding=padding, name=layer_name+'_output')
            log_tensor_info(output)
            if add_bias:
                dim = int(output.get_shape()[-1])
                bias = tf.get_variable(layer_name + '_bias', shape=[dim], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
                output = tf.nn.bias_add(output, bias)
            return output

    def _maxpool3d_layer(self, layer_name, input_data, kernel_size, stride, padding='VALID'):
        # Notification: kernel_size is like [1,2,2], stride is like [1,2,2]
        with tf.variable_scope(layer_name):
            kernel_shape = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
            stride_shape = [1, stride[0], stride[1], stride[2], 1]
            output = tf.nn.max_pool3d(input_data, ksize=kernel_shape, strides=stride_shape, padding=padding, name=layer_name+'_output')
            log_tensor_info(output)
            return output

    def _avgpool3d_layer(self, layer_name, input_data, kernel_size, stride, padding='VALID'):
        # Notification: kernel_size is like [1,2,2], stride is like [1,2,2]
        with tf.variable_scope(layer_name):
            kernel_shape = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
            stride_shape = [1, stride[0], stride[1], stride[2], 1]
            output = tf.nn.avg_pool3d(input_data, ksize=kernel_shape, strides=stride_shape, padding=padding,
                                      name=layer_name + '_output')
            log_tensor_info(output)
            return output

    def _relu_layer(self, layer_name, input_data):
        with tf.variable_scope(layer_name):
            output = tf.nn.relu(input_data)
            return output

    def _dropout_layer(self, layer_name, input_data):
        with tf.variable_scope(layer_name):
            output = tf.cond(self.ph_is_training, lambda: tf.nn.dropout(input_data, self.hparams.keep_prob), lambda: input_data)
            return output

    def _fc_layer(self, layer_name, input_data, hiddens, add_bias=True):
        with tf.variable_scope(layer_name):
            shape = input_data.get_shape().as_list()[1:]
            dims = reduce(lambda x, y: x * y, shape)
            input_data = tf.reshape(input_data, [-1, dims])
            kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
            kernel_shape = [int(input_data.get_shape()[1]), hiddens]
            kernel = self._variable(layer_name + '_kernel', shape=kernel_shape, initializer=kernel_init)
            output = tf.matmul(input_data, kernel, name=layer_name+'_output')
            log_tensor_info(output)
            if add_bias:
                bias = tf.get_variable(layer_name + '_bias', shape=[hiddens], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
                output = tf.nn.bias_add(output, bias)
            return output

    def _bn_layer(self, layer_name, input_data):
        with tf.variable_scope(layer_name):
            beta = tf.Variable(tf.constant(0.0, shape=[input_data.shape[-1]]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[input_data.shape[-1]]), name='gamma', trainable=True)
            axises = np.arange(len(input_data.shape) - 1).tolist()
            batch_mean, batch_var = tf.nn.moments(input_data, axises, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(self.ph_is_training, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            output = tf.nn.batch_normalization(input_data, mean, var, beta, gamma, 1e-3)
            return output
    ##########--Basic NN construction unit--end--##########