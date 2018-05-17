from __future__ import print_function
import cv2


def video_slice(video_abspath, slice_seconds):
    video_capture = cv2.VideoCapture()
    video_capture.open(video_abspath)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    nums = int(total_frames/(fps*slice_seconds) + 1)
    video_name_prefix = video_abspath.split('.')[0]
    video_slice_names = [video_name_prefix+'_'+str(number)+'.avi' for number in range(nums)]

    frames_counter = 0
    video_writer = None
    flag, img = video_capture.read()
    while flag:
        if frames_counter % (fps * slice_seconds) == 0:
            index = int(frames_counter / (fps * slice_seconds))
            video_slice_name = video_slice_names[index]
            # video_writer = cv2.VideoWriter(video_slice_name, cv2.FOURCC('M', 'J', 'P', 'G'), fps, size)
            video_writer = cv2.VideoWriter(video_slice_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

        frames_counter += 1
        video_writer.write(img)
        flag, img = video_capture.read()
    if not flag:
        print('{} {}'.format(total_frames, frames_counter))
    video_capture.release()

if __name__ == '__main__':
    video_abspath = '/home/ydp/Desktop/10.avi'
    slice_seconds = 60
    video_slice(video_abspath, slice_seconds)