from __future__ import division
import logging
import numpy as np
import cv2
import os
from smoke_detection_core.motion_detection import img_to_block
from train_and_detection.train_libs_auxiliary import join_path_my

def generate_valid_blocks(block_size, video_height, video_width, contours):
    mask = np.zeros([video_height, video_width, 3], np.uint8)
    cv2.fillPoly(mask, [np.array(contours, np.int32)], (255, 255, 255))
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    int_mask = cv2.integral(gray_mask)

    # This is a key parameter.
    block_diff = block_size * block_size * 30
    result = []
    locate_list = img_to_block(video_height, video_width, block_size)
    for location in locate_list:
        x, y, d1, d2 = location
        pixel_1 = int_mask[x, y]
        pixel_2 = int_mask[x + block_size, y]
        pixel_3 = int_mask[x, y + block_size]
        pixel_4 = int_mask[x + block_size, y + block_size]
        pixel_diff = pixel_4 - pixel_3 - pixel_2 + pixel_1
        if (pixel_diff >= block_diff):
            result.append([x, y])
    #         cv2.rectangle(mask, (y, x), (y + block_size, x + block_size), (0, 255, 0), 1)
    # cv2.imshow('img', mask)
    # cv2.waitKey(0)
    return result

def write_blocks_file(block_size, data_dir):
    labels_dir = os.path.join(data_dir, 'labels')
    blocks_files_dir = join_path_my(data_dir, 'blocks')

    video_labels_dirs = os.listdir(labels_dir)
    for video_name in video_labels_dirs:
        # Get basic info.
        video_name_avi = video_name + '.avi'
        video_path = os.path.join(data_dir, video_name_avi)
        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)
        if video_capture.isOpened():
            video_width = np.int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = np.int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_capture.release()
        else:
            logging.error('Video({}) is not opened! This video may be a bad video!'.format(video_path))
            continue

        per_video_blocks = video_name + '.txt'
        per_video_blocks_saved_file = os.path.join(blocks_files_dir, per_video_blocks)
        with open(per_video_blocks_saved_file, 'wb') as f:
            # write video_name
            f.write(bytes('% {}\n'.format(video_name_avi), encoding='utf-8'))

            labeled_frames_info = os.listdir(os.path.join(labels_dir, video_name))
            for frame_info in labeled_frames_info:
                def write_blocks(line):
                    contours = []
                    vertexes = line[1:-1].strip().split(',')
                    for vertex_str in vertexes:
                        vertex = vertex_str.strip().split(' ')
                        edge = [int(vertex[0]), int(vertex[1])]
                        contours.append(edge)
                    valid_blocks = generate_valid_blocks(block_size, video_height, video_width, contours)
                    if len(valid_blocks) > 0:
                        blocks_str = ['{} {}'.format(block[0], block[1]) for block in valid_blocks]
                        f.write(bytes('* {}\n'.format(','.join(blocks_str)), encoding='utf-8'))

                per_frame_info_intact_path = os.path.join(labels_dir, video_name, frame_info)
                # write frame index
                frame_index = frame_info.split('_')[0]
                f.write(bytes('# {}\n'.format(frame_index), encoding='utf-8'))
                with open(per_frame_info_intact_path, 'r') as info:
                    for line in info.readlines():
                        if line[0] == '*':
                            write_blocks(line)
