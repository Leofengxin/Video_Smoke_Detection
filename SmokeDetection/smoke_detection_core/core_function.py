from __future__ import print_function
import logging
import os
import shutil
import time
import cv2
import numpy as np
import tensorflow as tf
from data_prepare.generate_tfrecords import is_valid_frame_index
from smoke_detection_core.motion_detection import img_to_block
from win_libs.libs_auxiliary import VideoInfo

def smoke_classification(sess, model, frames_array, motion_blocks):
    # Use model to classify smoke.
    blocks_num = len(motion_blocks)
    smoke_blocks = []
    if blocks_num > 0:
        block_size = model.hparams.block_size
        all_block_data = np.zeros([blocks_num, model.hparams.sample_sum_frames, block_size, block_size, 3], dtype=np.uint8)
        for index, block in enumerate(motion_blocks):
            x, y = block[0], block[1]
            all_block_data[index, :, :, :, :] = frames_array[:, x:x + block_size, y:y + block_size, :]

        # Standardization. Keep coincident with training model.
        if model.hparams.is_standardization:
            all_block_data = (all_block_data - 128.0) / 128.0

        # Classify.
        argmax_labels = list()
        batch_num = model.hparams.batch_size
        batches = int(blocks_num/batch_num)
        for i in range(batches):
            batch_argmax_labels = sess.run(model.argmax_output,
                                   feed_dict={model.ph_data: all_block_data[i*batch_num:(i+1)*batch_num],
                                              model.ph_is_training: False})
            argmax_labels.append(batch_argmax_labels)
        if blocks_num%batch_num != 0:
            last_batch_data_start_index = batches*batch_num
            last_batch_argmax_labels = sess.run(model.argmax_output,
                                        feed_dict={model.ph_data: all_block_data[last_batch_data_start_index:],
                                                   model.ph_is_training: False})
            argmax_labels.append(last_batch_argmax_labels)
        argmax_labels = np.concatenate(argmax_labels, axis=0)
        smoke_blocks_indexes = np.where(argmax_labels==1)
        smoke_blocks = np.take(motion_blocks, smoke_blocks_indexes, axis=0)
        smoke_blocks = smoke_blocks[0]  # This code is added because smoke_blocks dimension is 3 when I debug.
    return smoke_blocks

def img_smoke_detection(sess, model, video_capture, video_info, location_list):
    _, interval_frame = is_valid_frame_index(model.hparams, video_info.frame_rate, 0)
    sample_sum_frames = model.hparams.sample_sum_frames
    block_size = model.hparams.block_size

    #####--For per frame, detect smoke--start--#####
    frames = []
    flag_TF, _ = is_valid_frame_index(model.hparams, video_info.frame_rate, video_info.frame_current)
    # If cureent_frame_index is valid.
    if flag_TF:
        # print(video_info.frame_current, sample_sum_frames)
        for i in range(sample_sum_frames):

            frame_idx = video_info.frame_current - (sample_sum_frames - i -1) * interval_frame

            #frame_idx = 200
            ##print(frame_idx)
            #video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            flag_f, cv2_img = video_capture.read()


            if flag_f:
                frames.append(cv2_img)
            else:
                logging.info('video_capture read video({}) in frame_index({}) failed,\
                 please check code and video.'.format(video_info.video_path, frame_idx))
                return [], []
        frames_array = np.array(frames)
        # Motion detection.
        motion_blocks = model.motion_detector(frames_array, location_list, block_size)
        # smoke_blocks = dark_channl(frames_array, location_list, block_size)
        # motion_blocks = location_list

        # Classify.
        smoke_blocks = smoke_classification(sess, model, frames_array, motion_blocks)
        # smoke_blocks = motion_blocks

        return smoke_blocks, motion_blocks
    else:
        return [], []
    #####--For per frame, detect smoke--end--#####

def videos_smoke_detection(videos_dir, ckpt_dir, model):
    # Clear old tfreocds.
    blocks_dir = os.path.join(videos_dir, 'blocks')
    if os.path.exists(blocks_dir):
        shutil.rmtree(blocks_dir)
    os.mkdir(blocks_dir)

    hard_videos = os.listdir(videos_dir)
    for idx, video in enumerate(hard_videos):
        if video.find('.avi') < 0:
            hard_videos.pop(idx)
    # def video_filter(name):
    #     return name.find('.avi') > -1
    # hard_videos = filter(video_filter, hard_videos)
    hard_videos_paths = [os.path.join(videos_dir, video_name) for video_name in hard_videos]

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=cfg)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    for idx, video_path in enumerate(hard_videos_paths):
        start_time = time.time()
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        video_info = VideoInfo()
        video_info.video_path = video_path
        video_info.frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        video_info.frame_total_num = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        video_info.frame_current = 0
        # This is a key parameter, to decide how long time to detect.
        interval_time = 0.3
        detection_interval = int(interval_time * video_info.frame_rate)
        # detection_interval = 0

        video_blocks_txt_file_name = hard_videos[idx].split('.')[0] + '.txt'
        video_blocks_txt_file_path = os.path.join(blocks_dir, video_blocks_txt_file_name)
        with open(video_blocks_txt_file_path, 'ab') as f:
            f.write(bytes('% {}\n'.format(hard_videos[idx]), encoding='utf-8'))

        rows = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cols = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        location_list = img_to_block(rows, cols, model.hparams.block_size)
        flag, img = video_capture.read()
        while flag:
            smoke_blocks, motion_block = img_smoke_detection(sess, model, video_capture, video_info, location_list)
            if len(smoke_blocks) > 0:
                with open(video_blocks_txt_file_path, 'ab') as f:
                    f.write(bytes('# {}\n'.format(video_info.frame_current), encoding='utf-8'))
                    blocks_str = ['{} {}'.format(block[0], block[1]) for block in smoke_blocks]
                    f.write(bytes('* {}\n'.format(','.join(blocks_str)), encoding='utf-8'))
            video_info.frame_current += detection_interval
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_info.frame_current)
            flag, img = video_capture.read()
        video_capture.release()
        duration = time.time() - start_time
        logging.info('Now detect video:{}, cost:{} s'.format(video_path, duration))
    sess.close()

def dark_channl(frames, location_list, block_size):
    r, g, b = cv2.split(frames[-1])
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dc_img = cv2.erode(min_img, kernel)

    ret, thresh1 = cv2.threshold(dc_img, 170, 255, cv2.THRESH_BINARY)

    cv2.imshow("dark_channl", thresh1)

    int_diff = cv2.integral(thresh1)
    # This is a key parameter. Change this value can control motion_block number.
    # threshold = block_size * block_size / 2
    # threshold = 400
    result = list()
    for pt in iter(location_list):
        xx, yy, _bz, _bz = pt
        t11 = int_diff[xx, yy]
        t22 = int_diff[xx + block_size, yy + block_size]
        t12 = int_diff[xx, yy + block_size]
        t21 = int_diff[xx + block_size, yy]
        block_diff = t11 + t22 - t12 - t21
        if block_diff > 0:
            result.append((xx, yy, block_size, block_size))
    return result



