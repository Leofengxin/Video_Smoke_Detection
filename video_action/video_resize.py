from __future__ import print_function
import os
import cv2


def video_resize(video_abspath):
    video_capture = cv2.VideoCapture()
    video_capture.open(video_abspath)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    index = video_abspath.rindex('/')
    video_path_resize = video_abspath[:index+1] + 'resize-' + video_abspath[index+1:]
    resized_size = (size[0]/2, size[1]/2)
    # video_writer = cv2.VideoWriter(video_path_resize, cv2.FOURCC('M', 'J', 'P', 'G'), fps, resized_size)
    video_writer = cv2.VideoWriter(video_path_resize, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, resized_size)

    frames_counter = 0
    flag, img = video_capture.read()
    while flag:
        frames_counter += 1
        img = cv2.resize(img, resized_size, interpolation=cv2.INTER_CUBIC)
        video_writer.write(img)
        flag, img = video_capture.read()
    if not flag:
        print ('total frames number is:{} but now frame is:{}!'.format(total_frames, frames_counter))
    video_capture.release()

if __name__ == '__main__':
    videos_dir = '/home/ydp/Desktop/videos'
    videos = os.listdir(videos_dir)
    videos = [os.path.join(videos_dir, name) for name in videos]
    for video in videos:
        video_resize(video)