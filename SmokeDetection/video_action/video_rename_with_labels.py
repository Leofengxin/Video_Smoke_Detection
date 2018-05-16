from __future__ import print_function
import os

if __name__ == '__main__':
    project_fir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(project_fir, 'data')
    dir_names = [(os.path.join(data_dir, 'test', 'positive'), 'test_pos_'),
                 (os.path.join(data_dir, 'test', 'negative'), 'test_neg_'),
                 (os.path.join(data_dir, 'train', 'positive'), 'train_pos_'),
                 (os.path.join(data_dir, 'train', 'negative'), 'train_neg_'),
                 (os.path.join(data_dir, 'hard_videos', 'negative'), 'hard_neg_')]

    for videos_dir, prefix in dir_names:
        all_ = os.listdir(videos_dir)
        files = []
        for file in all_:
            if os.path.isfile(os.path.join(videos_dir,file)):
                files.append(file)
        labels_dir = os.path.join(videos_dir, 'labels')
        for index, file in enumerate(files):
            old_video_name = os.path.join(videos_dir, file)
            new_video_name = os.path.join(videos_dir, '{}{}.avi'.format(prefix, index))
            os.rename(old_video_name, new_video_name)
            old_label_name = os.path.join(labels_dir, file.split('.')[0])
            new_label_name = os.path.join(labels_dir, '{}{}'.format(prefix, index))
            if os.path.exists(old_label_name):
                os.rename(old_label_name, new_label_name)
    print('Rename video and labels complete!')
