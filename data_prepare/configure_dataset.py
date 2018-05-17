from easydict import EasyDict

def config_dataset():
    conf = EasyDict()
    conf.block_size = 24
    conf.minimum_interval_frame = 1
    conf.minimum_interval_seconds = 0.01
    conf.sample_sum_frames = 8
    return conf
