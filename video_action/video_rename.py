from __future__ import print_function
import os

if __name__ == '__main__':
    project_fir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(project_fir, 'data')
    dir_names = [(os.path.join(data_dir, 'test', 'positive'), 'test_pos_'),
                 (os.path.join(data_dir, 'test', 'negative'), 'test_neg_'),
                 (os.path.join(data_dir, 'train', 'positive'), 'train_pos_'),
                 (os.path.join(data_dir, 'train', 'negative'), 'train_neg_')]

    for d, prefix in dir_names:
        files = os.listdir(d)
        for index, file in enumerate(files):
            old_name = os.path.join(d, file)
            new_name = os.path.join(d, prefix+str(index)+'.avi')
            os.rename(old_name, new_name)
    print('Rename complete!')
