import os
import tensorflow as tf

def read_tfrecords(conf, tfrecords_dir, train_or_test_option):
    all_files = os.listdir(tfrecords_dir)
    train_or_test_files = []
    for file in all_files:
        if file.startswith(train_or_test_option):
            train_or_test_files.append(file)
    valid_files = [os.path.join(tfrecords_dir, file) for file in train_or_test_files]

    tfrecords_data_queue = tf.train.string_input_producer(valid_files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecords_data_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    })

    label = tf.cast(features['label'], tf.int32)
    img_3d = tf.decode_raw(features['image'], tf.uint8)
    img_3d = tf.cast(img_3d, tf.float32)
    img_3d = tf.reshape(img_3d, [conf.sample_sum_frames, conf.block_size, conf.block_size, 3])

    if conf.is_standardization:
        img_3d = tf.div(tf.subtract(img_3d, tf.constant(128.0, dtype=tf.float32)), tf.constant(128.0, dtype=tf.float32))
    return img_3d, label