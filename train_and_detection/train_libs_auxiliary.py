import os
import time
import logging
import fileinput
from easydict import EasyDict
from NN_model.factory_provider import hparams_factory, model_factory
from data_prepare.configure_dataset import config_dataset

def get_model_and_hparams(net_is_using):
    # Get model hparams_conf.
    hparams_model = hparams_factory().get_model_hparams(net_is_using)
    # Get data hparams.
    hparams_data = config_dataset()
    # Merge hparams.
    hparams = merge_hparams(hparams_model, hparams_data)
    # Get model.
    model = model_factory().get_model(net_is_using, hparams)
    return hparams, model

def join_path_my(path, *paths):
    dst_dir = os.path.join(path, *paths)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    return dst_dir

def remove_and_new_file(f_path):
    if os.path.exists(f_path):
        os.remove(f_path)
    saved_file = open(f_path, 'wb')
    saved_file.close()

def get_dataset_num_hparams(saved_info_path):
    hparams_data_num = EasyDict()
    with open(saved_info_path, 'rb') as f:
        for line in f.readlines():
            line = str(line, encoding='utf-8')
            line_info = line.strip().split(':')
            if line_info[0] == 'train_positive_data_num':
                hparams_data_num.train_data_num = int(line_info[1])
            if line_info[0] == 'train_negative_data_num':
                hparams_data_num.train_data_num += int(line_info[1])
            if line_info[0] == 'test_positive_data_num':
                hparams_data_num.test_data_num = int(line_info[1])
            if line_info[0] == 'test_negative_data_num':
                hparams_data_num.test_data_num += int(line_info[1])
            if line_info[0] == 'hard_videos_negative_data_num':
                hparams_data_num.train_data_num += int(line_info[1])
    return hparams_data_num

def merge_hparams(hparams1, hparams2):
    hparams = EasyDict()
    for h1 in hparams1:
        hparams[h1] = hparams1[h1]
    for h2 in hparams2:
        hparams[h2] = hparams2[h2]
    return hparams

def save_hparams_and_generate_ckpt_dir(FLAGS, summary_dir, hparams):
    def save_hparams(dir):
        hparams_dir = join_path_my(dir,'hparams')
        hparams_file_name = 'hparams_'+str(len(os.listdir(hparams_dir))+1)+'.txt'
        hparams_file_path = os.path.join(hparams_dir, hparams_file_name)
        # Recorde the hyperparamters.
        with open(hparams_file_path, 'wb') as f:
            # Write current time
            current_time = time.asctime(time.localtime(time.time()))
            f.write(bytes(current_time + '\n', encoding='utf-8'))
            # Write hyperparameters about train
            f.write(bytes('data_tfrecords_dir:{}\n'.format(FLAGS.data_tfrecords_dir), encoding='utf-8'))
            f.write(bytes('net_is_using:{}\n'.format(FLAGS.net_is_using), encoding='utf-8'))
            f.write(bytes('is_continue_train:{}\n'.format(FLAGS.is_continue_train), encoding='utf-8'))
            f.write(bytes('continue_train_dir:{}\n'.format(FLAGS.continue_train_dir), encoding='utf-8'))
            f.write(bytes('train_hard_example_num:{}\n'.format(FLAGS.train_hard_num), encoding='utf-8'))
            f.write(bytes('summary_step:{}\n'.format(str(FLAGS.summary_step)), encoding='utf-8'))
            f.write(bytes('checkpoint_step:{}\n'.format(str(FLAGS.checkpoint_step)), encoding='utf-8'))
            # Write hyperparameters about dataset and model
            for e in sorted(hparams.items(), key=lambda d: d[0]):
                f.write(bytes('{}:{}\n'.format(e[0], e[1]), encoding='utf-8'))
            f.write(bytes('--------------------------------\n', encoding='utf-8'))
            f.write(bytes('--------------------------------\n', encoding='utf-8'))
        return hparams_file_path
    def get_max_index():
        dirs = os.listdir(summary_dir)
        max_index = 0
        for dir in dirs:
            parts = dir.split('_')
            index = int(parts[-1])
            if index > max_index:
                max_index = index
        return max_index

    if FLAGS.continue_train_dir and FLAGS.is_continue_train:
        hparams_file_path = save_hparams(FLAGS.continue_train_dir)
        logging.info('Continue to train, the ckpt_dir:{}'.format(FLAGS.continue_train_dir))
        logging.info('Continue to train, the new hparams_file_path:{}'.format(hparams_file_path))
        return FLAGS.continue_train_dir, hparams_file_path
    else:
        max_index = get_max_index()
        dir_name = FLAGS.net_is_using + '_' + str(max_index + 1)
        ckpt_dir = join_path_my(summary_dir, dir_name)
        hparams_file_path = save_hparams(ckpt_dir)
        logging.info('Train a new model, the ckpt_dir:{}'.format(ckpt_dir))
        logging.info('Train a new model, the hparams_file_path:{}'.format(hparams_file_path))
        return ckpt_dir, hparams_file_path

def write_test_acc(hparams_file_path, epoch, global_step, acc):
    with open(hparams_file_path, 'ab') as f:
        info = 'Epoch:{}  global_step:{}  acc:{}\n'.format(epoch, global_step, acc)
        f.write(bytes(info, encoding='utf-8'))
        logging.debug(info)

def keep_best_ckpt(ckpt_dir, top_k_acc):
    acc = [test_acc[1] for test_acc in top_k_acc]
    max_acc = max(acc)
    max_idx = acc.index(max_acc)
    max_ele = top_k_acc[max_idx]
    first_line = ('model_checkpoint_path: "{}"'.format(max_ele[0]))
    for line in fileinput.input(os.path.join(ckpt_dir, 'checkpoint'), inplace=True):
        if fileinput.isfirstline():
            line = first_line
            print(line.strip())
        else:
            print(line.strip())
    with open(os.path.join(ckpt_dir, 'checkpoint_my.txt'), 'wb') as f:
        for ele in top_k_acc:
            path, acc = ele
            f.write(bytes('{}:{}\n'.format(path, acc), encoding='utf-8'))

def save_ckpt_file(sess, saver, ckpt_dir, global_step, top_k_acc, test_accuracy):
    ckpt_path = os.path.join(ckpt_dir, 'model.ckpt')
    saver.save(sess, ckpt_path, global_step=global_step)
    ckpt_path = ckpt_path + '-' + str(global_step)
    ckpt_dir_my = os.path.join(ckpt_dir, 'checkpoint_my.txt')
    if os.path.exists(ckpt_dir_my):
        f = open(ckpt_dir_my, 'rb')
        for line in f.readlines():
            line = str(line, encoding='utf-8')
            line_info = line.strip().split(':')
            path = line_info[0].strip()
            acc = float(line_info[1])
            ele = [path, acc]
            if ele not in top_k_acc:
                top_k_acc.append(ele)
        f.close()
    if len(top_k_acc) < 10:
        top_k_acc.append([ckpt_path, test_accuracy])
        keep_best_ckpt(ckpt_dir, top_k_acc)
    else:
        acc = [test_acc[1] for test_acc in top_k_acc]
        min_acc = min(acc)
        if test_accuracy > min_acc:
            top_k_acc.append([ckpt_path, test_accuracy])
            min_idx = acc.index(min_acc)
            min_ele = top_k_acc.pop(min_idx)
            basename = os.path.basename(min_ele[0])
            ckpt_files = os.listdir(ckpt_dir)
            for ckpt_file in ckpt_files:
                if ckpt_file.startswith(basename):
                    os.remove(os.path.join(ckpt_dir, ckpt_file))
                    # print('remove({})'.format(os.path.join(ckpt_dir, ckpt_file)))
            for line in fileinput.input(os.path.join(ckpt_dir, 'checkpoint'), inplace=True):
                if line.find(basename) > -1:
                    pass
                else:
                    print(line.strip())
            keep_best_ckpt(ckpt_dir, top_k_acc)
        else:
            keep_best_ckpt(ckpt_dir, top_k_acc)