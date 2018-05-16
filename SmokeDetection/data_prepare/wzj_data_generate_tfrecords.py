import logging
from logging_core.logging_setup import setup_logging
setup_logging()
import os
import time
import cv2
import fileinput
import numpy as np
import tensorflow as tf
from data_prepare.configure_dataset import config_dataset
from train_and_detection.train_libs_auxiliary import join_path_my, remove_and_new_file

def is_valid_frame_index(conf, frame_rate, current_frame_index):
    interval_frame = int(conf.minimum_interval_seconds * frame_rate)
    if interval_frame < conf.minimum_interval_frame:
        interval_frame = conf.minimum_interval_frame
    valid_frame_index = conf.sample_sum_frames * interval_frame
    # The judging condition is being little problem. But is not very important.
    if current_frame_index < valid_frame_index:
        return False, interval_frame
    else:
        return True, interval_frame

def one_sample_write(tfrecords_writer, one_sample_img, one_sample_label):
    one_sample_img_raw = np.uint8(one_sample_img)
    one_sample_img_str = one_sample_img_raw.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[one_sample_label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[one_sample_img_str]))
    }))
    tfrecords_writer.write(example.SerializeToString())

def generate_tfrecords_writer(data_dir, tfrecords_writer, num, writer_option, pos_flag):
    # This is a key parameter.
    one_tfrecords_num = 50
    if num % one_tfrecords_num == 0:
        tfrecords_file_index = num / one_tfrecords_num
        tfrecords_data_dir = join_path_my(data_dir, 'data_tfrecords')
        tfrecords_filename = '{}_{}_data_{}.tfrecords'.format(writer_option, pos_flag, tfrecords_file_index)
        data_tfrecords_path = os.path.join(tfrecords_data_dir, tfrecords_filename)
        if num !=0:
            tfrecords_writer.close()
        tfrecords_writer = tf.python_io.TFRecordWriter(data_tfrecords_path)
    return tfrecords_writer

def tfrecords_data_write(data_dir, conf, video_capture, current_frame_index, valid_blocks, pos_flag,
                         writer_option, tfrecords_writer, data_num):
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    _, interval_frame = is_valid_frame_index(conf, frame_rate, current_frame_index)

    frames = []
    for i in range(conf.sample_sum_frames):
        frame_idx = current_frame_index - (conf.sample_sum_frames - i -1) * interval_frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        flag, cv2_img = video_capture.read()
        if flag:
            frames.append(cv2_img)
        else:
            logging.error('Currently video_capture read video in frame_index({}) failed, please pay attention to this video!'.format(frame_idx))
            return tfrecords_writer, data_num
    frames_array = np.array(frames)
    for block in valid_blocks:
        x, y = block[0], block[1]
        one_sample_img = frames_array[:, x:x+conf.block_size, y:y+conf.block_size, :]
        one_sample_label = 1 if pos_flag == 'positive' else 0
        tfrecords_writer = generate_tfrecords_writer(data_dir, tfrecords_writer, data_num, writer_option, pos_flag)
        one_sample_write(tfrecords_writer, one_sample_img, one_sample_label)
        data_num += 1
    return tfrecords_writer, data_num

def generate_tfrecords_core(conf, data_root_dir, tfrecords_writer, data_num, writer_option, pos_flag):
    data_dir = os.path.join(data_root_dir, writer_option, pos_flag)

    video_capture = cv2.VideoCapture()
    videos = os.listdir(data_dir)
    for video in videos:
        video_path = os.path.join(data_dir, video)
        video_capture.open(video_path)
        video_frames_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame_index = 7
        flag, img = video_capture.read()
        while flag:
            valid_blocks = [[0,0]]
            flag, _ = video_capture.read()
            tfrecords_writer, data_num = tfrecords_data_write(data_root_dir, conf, video_capture,
                                                              current_frame_index, valid_blocks, pos_flag,
                                                              writer_option, tfrecords_writer, data_num)
            current_frame_index += conf.sample_sum_frames//2
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        if current_frame_index == video_frames_count:
            logging.info('video({}) has handled!'.format(video_path))
        else:
            logging.error('video({}) encounter a problem in frame({}).'.format(video_path, current_frame_index))
        if video_capture.isOpened():
            video_capture.release()

    return tfrecords_writer, data_num

def generate_tfrecords(conf, data_dir, writer_option, pos_flag):
    #####--Generate tfrecords file.--start--#####
    start_time = time.time()

    # Generate train_data_tfrecords.
    data_num = 0
    tfrecords_writer = None
    tfrecords_writer = generate_tfrecords_writer(data_dir, tfrecords_writer, data_num, writer_option, pos_flag)
    tfrecords_writer, data_num = generate_tfrecords_core(conf, data_dir, tfrecords_writer, data_num, writer_option, pos_flag)
    tfrecords_writer.close()

    # Write samples number.
    saved_file_path = os.path.join(data_dir, 'dataset_info.txt')
    writed_flag = False
    for line in fileinput.input(saved_file_path, inplace=True):
        if line.startswith('{}_{}'.format(writer_option, pos_flag)):
            line = '{}_{}_data_num:{}'.format(writer_option, pos_flag, data_num)
            print(line.strip())
            writed_flag = True
        else:
            print(line.strip())
    if not writed_flag:
        with open(saved_file_path, 'ab') as f:
            f.write(bytes('{}_{}_data_num:{}\n'.format(writer_option, pos_flag, data_num), encoding='utf-8'))

    duration = time.time() - start_time
    logging.info('{}_{}_data_num:{}'.format(writer_option, pos_flag, data_num))
    logging.info('Generate {}_{} tfrecords duration:{}'.format(writer_option, pos_flag, duration))
    #####--Generate tfrecords file.--end--#####

def generate_tfrecords_first_time():
    # Dataset configure.
    conf = config_dataset()
    conf.blcok_size = 24
    conf.minimum_interval_frame = 1
    conf.minimum_interval_seconds = 0.01
    conf.sample_sum_frames = 8

    # Generate blocks and write to file.
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')

    saved_file_path = os.path.join(data_dir, 'dataset_info.txt')
    remove_and_new_file(saved_file_path)
    generate_tfrecords(conf, data_dir, 'train', 'positive')
    generate_tfrecords(conf, data_dir, 'train', 'negative')
    generate_tfrecords(conf, data_dir, 'test', 'positive')
    generate_tfrecords(conf, data_dir, 'test', 'negative')

if __name__ == '__main__':
    generate_tfrecords_first_time()
