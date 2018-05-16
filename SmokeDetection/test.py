# from __future__ import print_function
# import os
# import fileinput
#
# project_dir = os.path.dirname(os.path.abspath(os.getcwd()))
# summary_dir = os.path.join(project_dir, 'summary')
# ckpt_dir = os.path.join(summary_dir, 'cnn3d_1')
# ckpt_file_path = os.path.join(ckpt_dir, 'checkpoint')
#
# for line in fileinput.input(ckpt_file_path, inplace=True):
#     if line.startswith('Fri'):
#         line = 'sd'
#         print(line.strip())
#     else:
#         print(line.strip())

from __future__ import print_function
import os

with open('ss.txt', 'ab') as f:
    f.write(bytes('sddd', encoding='utf-8'))