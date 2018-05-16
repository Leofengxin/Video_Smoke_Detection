from __future__ import division, print_function
import logging
from logging_core.logging_setup import setup_logging
setup_logging()
import os
import time
import numpy as np
import tensorflow as tf
from train_and_detection.train_libs_auxiliary import join_path_my, get_dataset_num_hparams, merge_hparams, write_test_acc
from train_and_detection.train_libs_auxiliary import get_model_and_hparams, save_hparams_and_generate_ckpt_dir, save_ckpt_file
from data_prepare.generate_tfrecords import generate_tfrecords_first_time, generate_tfrecords
from data_prepare.read_tfrecords import read_tfrecords
from smoke_detection_core.core_function import videos_smoke_detection

def train(model, hparams, ckpt_dir, hparams_file_path, data_and_labels):
    # Data and labels.
    train_batch_data, train_batch_label, test_batch_data, test_batch_label = data_and_labels

    # Create session and save graph.
    sess = tf.InteractiveSession()
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)

    # Restore variables, train continually or initialize all.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        trained_steps = int(ckpt.model_checkpoint_path.split('-')[-1])
        # The following code is for tf.train.string_input_producer(), where define num_epochs parameter.
        # sess.run(tf.local_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
        trained_steps = 0
        # The following code is for tf.train.string_input_producer(), where define num_epochs parameter.
        # sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        # There, k is default to 10.
        top_k_acc = []
        for epoch in range(hparams.epoches):
            #####--PartI: Model train and test. --start--#####
            def train_one_step():
                # Feed dict, train phase.
                feed_data, feed_label = sess.run([train_batch_data, train_batch_label])
                feed_dict = {model.ph_data: feed_data, model.ph_label: feed_label,
                             model.ph_is_training: True}

                # For per summary_step, print current train acc, otherwise just train a batch_size data.
                if step % FLAGS.summary_step == 0 and step != 0:
                    op_list = [model.train_op, model.argmax_output, summary_op]
                    _, train_out_label, summaries = sess.run(op_list, feed_dict=feed_dict)

                    # Summary flush.
                    summary_writer.add_summary(summaries, global_step=global_step)
                    summary_writer.flush()

                    # predict_correct_num = tf.nn.in_top_k(np.argmax(probs, axis=1), feed_label)
                    train_std_label = np.argmax(feed_label, axis=1)
                    predict_correct_num = np.sum(train_out_label == train_std_label)
                    batch_accuracy = predict_correct_num / hparams.batch_size
                    train_acc.append(batch_accuracy)

                    # Output labels and train_acc.
                    avg_train_accuracy = np.mean(train_acc[-FLAGS.summary_step:-1])
                    print('{}\n{} | train_acc: {} {} {}'.format(train_std_label, train_out_label, batch_accuracy, avg_train_accuracy, global_step))
                else:
                    _, train_out_label = sess.run([model.train_op, model.argmax_output], feed_dict=feed_dict)
                    train_std_label = np.argmax(feed_label, axis=1)
                    predict_correct_num = np.sum(train_out_label == train_std_label)
                    batch_accuracy = predict_correct_num / hparams.batch_size
                    train_acc.append(batch_accuracy)
            def test_ckpt(current_epoch, global_step):
                start_time_test = time.time()
                test_steps = int(hparams.test_data_num / hparams.batch_size)
                sum = 0
                for test_step in range(test_steps):
                    test_feed_data, test_feed_label = sess.run([test_batch_data, test_batch_label])
                    output_label = sess.run(model.argmax_output, feed_dict={model.ph_data: test_feed_data, model.ph_is_training: False})
                    standard_label = np.argmax(test_feed_label, axis=1)
                    sum += np.sum(output_label == standard_label)

                    # Output labels and batch_accuracy.
                    if test_step % FLAGS.summary_step == 0 and test_step != 0:
                        batch_accuracy = np.sum(output_label == standard_label) / hparams.batch_size
                        print('{}\n{} | batch_acc: {}'.format(standard_label, output_label, batch_accuracy))
                test_accuracy = sum / (test_steps * hparams.batch_size)

                print('test acc:{} {}'.format(test_accuracy, global_step))
                duration = time.time() - start_time_test
                print('Test in {} epoch cost {}'.format(current_epoch, duration))
                logging.error('test acc:{} {}'.format(test_accuracy, global_step))
                return test_accuracy

            start_time_epoch = time.time()
            steps = int(hparams.train_data_num / hparams.batch_size)
            train_acc = []
            for step in range(steps):
                # Global step and global epoches.
                global_step = trained_steps + epoch * steps + step
                # current_epoch = global_step * hparams.batch_size // hparams.train_data_num

                # Train one step.
                train_one_step()

                # For per checkpoint_step, calculate and print acc in test data.
                if step % FLAGS.checkpoint_step == 0 and step != 0:
                    # Test.
                    test_accuracy = test_ckpt(epoch, global_step)
                    # Write the test_accuracy to file
                    write_test_acc(hparams_file_path, epoch, global_step, test_accuracy)

                    # Save ckpt.
                    save_ckpt_file(sess, saver, ckpt_dir, global_step, top_k_acc, test_accuracy)

            duration = time.time() - start_time_epoch
            print('The {} epoch duration: {}'.format(epoch, duration))
            #####--PartI: Model train and test. --end--#####

        #####--PartII: Video smoke_detection_win.--start--#####
        # smoke_detection_win(sess, model)
        #####--PartII: Video smoke_detection_win.--end--#####
    except Exception as e:
        logging.exception(e)
    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()

def train_hard():
    # Get model and hparams.
    hparams, model = get_model_and_hparams(FLAGS.net_is_using)

    # Save hparams and get summary directory, where ckpt file is wrote.
    ckpt_dir, hparams_file_path = save_hparams_and_generate_ckpt_dir(FLAGS, summary_dir, hparams)

    # Get tfrecods_dir.
    data_dir = os.path.join(project_dir, 'data')
    tfrecords_dir = FLAGS.data_tfrecords_dir
    for i in range(FLAGS.train_hard_num):
        # If this is the first time to train, and not exist tfrecords_dir, generate train and test data.
        # Else clear existing hard_examples .tfrecords and do smoke detection, generate hard_examples .tfrecords.
        if i == 0 and len(os.listdir(tfrecords_dir)) == 0:
            generate_tfrecords_first_time()
        elif i != 0:
            # Clear existing hard examples .tfrecords.
            tfrecords_files = os.listdir(tfrecords_dir)
            for file in tfrecords_files:
                if file.startswith('hard_videos'):
                    os.remove(os.path.join(tfrecords_dir, file))
            # Smoke detection.
            hard_videos_dir = os.path.join(data_dir, 'hard_videos', 'negative')
            videos_smoke_detection(hard_videos_dir, ckpt_dir, model)
            # Generate hard examples to .tfrecords.
            generate_tfrecords(hparams, data_dir, 'hard_videos', 'negative')

        # Add train_data_num and test_data_num hparams.
        saved_info_path = os.path.join(data_dir, 'dataset_info.txt')
        hparams_data_num = get_dataset_num_hparams(saved_info_path)
        hparams = merge_hparams(hparams, hparams_data_num)

        # Prepare train data.
        train_raw_data, train_raw_label = read_tfrecords(hparams, tfrecords_dir, ('train', 'hard_videos'))
        train_label = tf.sparse_to_dense(sparse_indices=[train_raw_label], output_shape=[hparams.num_classes],
                                         sparse_values=1.0, )
        train_batch_data, train_batch_label = tf.train.shuffle_batch([train_raw_data, train_label],
                                                                     batch_size=hparams.batch_size,
                                                                     capacity=hparams.batch_size * 50,
                                                                     min_after_dequeue=hparams.batch_size * 10,
                                                                     num_threads=2, allow_smaller_final_batch=False)
        # Prepare est data.
        test_raw_data, test_raw_label = read_tfrecords(hparams, tfrecords_dir, ('test'))
        test_label = tf.sparse_to_dense(sparse_indices=[test_raw_label], output_shape=[hparams.num_classes],
                                        sparse_values=1.0, )
        test_batch_data, test_batch_label = tf.train.shuffle_batch([test_raw_data, test_label],
                                                                   batch_size=hparams.batch_size,
                                                                   capacity=hparams.batch_size * 50,
                                                                   min_after_dequeue=hparams.batch_size * 10,
                                                                   num_threads=2, allow_smaller_final_batch=False)
        data_and_labels = [train_batch_data, train_batch_label, test_batch_data, test_batch_label]

        # Train the model.
        train(model, hparams, ckpt_dir, hparams_file_path, data_and_labels)

# Summary dir and tfrecords data dir.
project_dir = os.path.dirname(os.getcwd())
summary_dir = join_path_my(project_dir, 'summary')
data_tfrecords_dir = join_path_my(project_dir, 'data', 'data_tfrecords')
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_tfrecords_dir', data_tfrecords_dir, 'Where tfrecords data locate.')
tf.app.flags.DEFINE_string('net_is_using', 'cnn3d', 'The net is in using now. Option ["cnn3d", "cnn2d_lstm"].')
tf.app.flags.DEFINE_string('motion_detector', 'background_substraction', 'The motion detector is using now. Option ["background_substraction"]')
tf.app.flags.DEFINE_bool('is_continue_train', False, 'Whether or not to train continually.')
tf.app.flags.DEFINE_string('continue_train_dir', os.path.join(summary_dir, 'cnn3d_16'), 'The directory where model is saved.')
tf.app.flags.DEFINE_integer('train_hard_num', 2, 'Number of training hard example')
tf.app.flags.DEFINE_integer('summary_step', 10, 'Number of steps to save summary.')
tf.app.flags.DEFINE_integer('checkpoint_step', 20, ' Number of steps to save summary.')

# Notation: There are several key parameters need to set.
# First, block_diff to decide where the block is smoke_block.
#        data_prepare/generate_blocks.py/generate_valid_blocks()
#        block_diff = block_size * block_size * 30
# Second, one_tfrecords_num to set the number of a .tfreocds file.
#        data_prepare/generate_tfrecords.py/generate_tfrecords_writer()
#        one_tfrecords_num = 300
# Third, threshold to decide whether a block is a motion_block.
#        train_libs_auxiliary.py/motion_detection()
# Forth, interval_time to decide how long detect the image's smoke blocks.
#        smoke_detection_core/videos_smoke_detection()
# Some parameters which are available to change.
# First, top_k_acc to save ckpt files, default to save top-10 best model.
#        train()
# Second, train_data_num and test_data_num parameters is saved to a .txt file
#        this file is default to data/dataset_info.txt
# Some operations need to attend.
# First, data standardization.
#        data_prepare/read_tfrecords/read_tfrecords()
#        smoke_detection_core/core_function/smoke_classification()
if __name__ == '__main__':
    train_hard()