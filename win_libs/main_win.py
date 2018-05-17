import logging
import os
from functools import partial
import cv2
import tensorflow as tf
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from smoke_detection_core.motion_detection import img_to_block
from smoke_detection_core.core_function import img_smoke_detection
from train_and_detection.train_libs_auxiliary import get_model_and_hparams
from win_libs.canvas import Canvas
from win_libs.libs_auxiliary import newIcon, newAction, struct, addActions, VideoInfo, cv2_img_to_qt_pixmap
from win_libs.toolbar import ToolBar


class MainWindow(QMainWindow):
    def __init__(self, sess=None, model=None):
        super(MainWindow, self).__init__()
        self.setMinimumSize(QSize(1200, 600))
        self.sess = sess
        self.model = model

        #####--Layout--start--#####
        #--left--#
        self.video_selection_label = QLabel('Videos')
        self.video_filelist = QListWidget()
        self.video_filelist.setFixedWidth(180)
        left_vbox = QVBoxLayout()
        left_vbox.addWidget(self.video_selection_label)
        left_vbox.addWidget(self.video_filelist)
        left_container = QWidget()
        left_container.setLayout(left_vbox)
        self.file_dock = QDockWidget()
        self.file_dock.setWidget(left_container)
        #--central--#
        self.canvas = Canvas()
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal:scroll.horizontalScrollBar()
        }
        #--right--#
        self.frame_rate_label = QLabel('Frame rate')
        self.frame_rate_value = QLabel('1')
        self.frame_total_label = QLabel('Frame total num')
        self.frame_total_value = QLabel('0')
        self.smoke_frame_num_label = QLabel('Smoke frame num')
        self.smoke_frame_num_value = QLabel('0')
        self.frame_current_label = QLabel('Frame current')
        self.frame_current_value = QSpinBox()
        self.frame_current_value.setMinimum(0)
        self.frame_current_value.setSingleStep(1)
        self.frame_current_value.setValue(0)
        self.frame_current_value.setEnabled(False)
        self.video_play_pause_button = QPushButton()
        self.video_play_pause_button.setCheckable(True)
        self.video_play_pause_button.setIcon(newIcon('play'))
        self.video_play_pause_button.setFixedWidth(30)
        self.video_play_pause_button.setEnabled(False)
        self.video_current_time_label = QLabel('0:0:0')
        self.video_total_time_label = QLabel('0:0:0')
        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.video_play_pause_button)
        hbox_layout.addWidget(self.video_current_time_label)
        hbox_layout.addWidget(self.video_total_time_label)
        hbox_container = QWidget()
        hbox_container.setLayout(hbox_layout)
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(1000)
        self.video_slider.setPageStep(0)
        self.video_slider.setEnabled(False)
        right_layout = QGridLayout()
        right_layout.addWidget(self.frame_rate_label, 0, 0)
        right_layout.addWidget(self.frame_rate_value, 0, 1)
        right_layout.addWidget(self.frame_total_label, 1, 0)
        right_layout.addWidget(self.frame_total_value, 1, 1)
        right_layout.addWidget(self.smoke_frame_num_label, 2, 0)
        right_layout.addWidget(self.smoke_frame_num_value, 2, 1)
        right_layout.addWidget(self.frame_current_label, 3, 0)
        right_layout.addWidget(self.frame_current_value, 3, 1)
        right_layout.addWidget(hbox_container, 4, 0, 1, 2)
        right_layout.addWidget(self.video_slider, 5, 0, 1, 2)
        video_info_container = QWidget()
        video_info_container.setLayout(right_layout)
        video_info_container.setFixedHeight(200)
        self.model_selection_label = QLabel('Models')
        self.model_filelist = QListWidget()
        self.model_filelist.setFixedWidth(180)
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(video_info_container)
        right_vbox.addWidget(self.model_selection_label)
        right_vbox.addWidget(self.model_filelist)
        right_container = QWidget()
        right_container.setLayout(right_vbox)
        self.video_dock = QDockWidget()
        self.video_dock.setWidget(right_container)
        #--layout--#
        # Set the central widget.
        self.setCentralWidget(scroll)
        # Set the dock areas for dock widget.
        self.file_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.file_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.video_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.video_dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.video_dock)
        #####--Layout--end--#####

        #####--Widget slot--start--#####
        self.model_filelist.doubleClicked.connect(self.model_filelist_doubleclicked)
        self.video_filelist.doubleClicked.connect(self.video_filelist_doubleclicked)
        self.frame_current_value.valueChanged.connect(self.frame_current_value_changed)
        self.video_play_pause_button.toggled.connect(self.video_play_pause_button_changed)
        self.video_slider.sliderPressed.connect(self.video_slider_pressed_respond)
        self.video_slider.sliderMoved.connect(self.video_slider_moved_respond)
        self.video_slider.sliderReleased.connect(self.video_slider_released_respond)
        #####--Widget slot--end--#####

        ##### --Toolbar and toolbar action--start-- #####
        p_action = partial(newAction, self)
        open_video = p_action('Open video', self.a_open_video, 'Ctrl+a', 'open', 'Open video file.')
        open_model = p_action('Open model', self.a_open_model, 'Ctrl+s', 'open', )
        open_next_video = p_action('Next video', self.a_open_next_video, 'Ctrl+x', 'next', 'Open next video.', enabled=False)
        open_prev_video = p_action('Prev video', self.a_open_prev_video, 'Ctrl+c', 'prev', 'Open prev video.', enabled=False)
        motion_block_hide = p_action('Hide motion block', self.a_hide_motion_block, 'h+m', 'hide', 'Hide motion block.')
        smoke_block_hide = p_action('Hide smoke block', self.a_hide_smoke_block, 'h+s', 'hide', 'Hide smoke block.')

        self.tool_bar = ToolBar('ToolBar')
        self.actions = struct(open_video=open_video,
                              open_model=open_model,
                              open_next_video=open_next_video,
                              open_prev_video=open_prev_video,
                              motion_block_hide=motion_block_hide,
                              smoke_block_hide=smoke_block_hide)
        addActions(self.tool_bar, (open_video, open_model, None,
                                   open_next_video, open_prev_video, None,
                                   motion_block_hide, smoke_block_hide))
        self.addToolBar(Qt.LeftToolBarArea, self.tool_bar)
        ##### --Toolbar and toolbar action--end-- #####
        ##### --Initial state--start-- #####
        self.timer_main = QTimer(self)
        self.timer_main.timeout.connect(self.main_timeout_respond)
        self.video_capture = cv2.VideoCapture()
        self.video_filter = ['.avi', '.mp4']
        self.video_info = VideoInfo()
        self.video_file_dir = os.getcwd()
        self.video_filename_current = 'canvas_pic.jpeg'
        self.video_filename = []
        self.smoke_frame_num = 0
        self.model_name_current = ''
        self.models = []
        if self.sess is None:
            self.sess = tf.InteractiveSession()
            self.phase = 'test'
        else:
            self.phase = 'train'
            self.actions.open_model.setEnabled(False)
            self.model_selection_label.hide()
            self.model_filelist.hide()
        self.load_video()
        ##### --Initial state--end-- #####

    ##### --Button action--start-- #####
    def a_open_video(self):
        filters = 'Video files (*{})'.format(' *'.join(self.video_filter))
        filename = QFileDialog.getOpenFileName(self, 'Choose video file.', filter=filters)
        if filename[0]:
            self.load_filename(filename)

    def a_open_model(self):
        model = QFileDialog.getExistingDirectory(self, 'Choose model')
        if model != '':
            models_dir = os.path.dirname(model)
            models = os.listdir(models_dir)
            models.sort()
            self.model_name_current = model
            self.model_filelist.addItems(models)
            self.models = [os.path.join(models_dir, model) for model in models]
            self.load_model(self.model_name_current)

    def a_open_next_video(self):
        idx_cur = self.video_filename.index(self.video_filename_current)
        idx = idx_cur + 1
        if idx >= len(self.video_filename):
            return
        # Update video_filelist.
        self.video_filename_current = self.video_filename[idx]
        self.video_filelist.setCurrentRow(idx)
        self.load_video()

    def a_open_prev_video(self):
        idx_cur = self.video_filename.index(self.video_filename_current)
        idx = idx_cur - 1
        if idx < 0:
            return
        # Update video_filelist.
        self.video_filename_current = self.video_filename[idx]
        self.video_filelist.setCurrentRow(idx)
        self.load_video()

    def a_hide_motion_block(self):
        self.canvas.is_hide_motion_blocks = not self.canvas.is_hide_motion_blocks
        self.canvas.repaint()

    def a_hide_smoke_block(self):
        self.canvas.is_hide_smoke_blocks = not self.canvas.is_hide_smoke_blocks
        self.canvas.repaint()
    ##### --Button action--end-- #####

    #####--Widget slot--start--#####
    def video_filelist_doubleclicked(self):
        idx_cur = self.video_filelist.currentRow()
        if self.video_filename_current == self.video_filename[idx_cur]:
            return
        else:
            self.video_filename_current = self.video_filename[idx_cur]
            self.load_video()

    def model_filelist_doubleclicked(self):
        idx_cur = self.model_filelist.currentRow()
        model_name_current = self.models[idx_cur].split('/')[-1]
        if self.model_name_current == model_name_current:
            return
        else:
            tf.reset_default_graph()
            if self.phase == 'test':
                self.sess.close()
                cfg = tf.ConfigProto()
                cfg.gpu_options.allow_growth = True
                self.sess = tf.InteractiveSession(config=cfg)
            self.model_name_current = model_name_current
            self.load_model(self.models[idx_cur])

    def frame_current_value_changed(self):
        self.video_info.frame_current = self.frame_current_value.value()
        self.update_video_info()
        #self.repaint_canvas()#####################################################################################################################
        self.timer_main.stop()

    def video_play_pause_button_changed(self):
        if self.video_play_pause_button.isChecked():
            self.timer_main.stop()
            self.video_play_pause_button.setIcon(newIcon('pause'))
            self.frame_current_value.setEnabled(True)
        else:
            self.timer_main.start(1000 / self.video_info.frame_rate)
            self.video_play_pause_button.setIcon(newIcon('play'))
            self.frame_current_value.setEnabled(False)


    def video_slider_pressed_respond(self):
        self.timer_main.stop()
        self.video_play_pause_button.setIcon(newIcon('pause'))
        self.video_play_pause_button.setChecked(True)
        self.frame_current_value.setEnabled(True)

    def video_slider_moved_respond(self):
        return

    def video_slider_released_respond(self):
        # self.video_play_pause_button.setChecked(False)
        self.video_info.frame_current = int(
            float(self.video_info.frame_total_num - 1) * 0.001 * self.video_slider.value())
        self.update_video_info()
        #self.repaint_canvas()#################################################################################################################
        self.timer_main.stop()

    def main_timeout_respond(self):
        # Update video_info.
        self.video_info.frame_current += 1
        self.update_video_info()
        # Update canvas
        self.repaint_canvas()

    def closeEvent(self, QCloseEvent):
        if self.phase == 'test':
            self.sess.close()
        else:
            pass
    #####--Widget slot--end--#####

    ##### --Auxiliary function--start-- #####
    def load_model(self, model_dir):
        model_name = model_dir.split('/')[-1]
        index = self.models.index(model_dir)
        self.model_filelist.setCurrentRow(index)
        model_name = '_'.join(model_name.split('_')[:-1])
        # Get hparams and model.
        _, self.model = get_model_and_hparams(model_name)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            logging.info('Restore model successfully!')
        else:
            logging.error('Can not restore model. Please check again!')
            return

    def load_video(self):
        if len(self.video_filename) == 0:
            default_img = QImage(os.path.join(self.video_file_dir, self.video_filename_current))
            qt_pixmap = QPixmap(default_img)
            self.canvas.load_pixmap(qt_pixmap)
        else:
            # Release the old video_capture.
            if self.video_capture.isOpened():
                self.video_capture.release()
            video_path = os.path.join(self.video_file_dir, self.video_filename_current)
            self.video_capture.open(video_path)

            # Update video_info.
            self.smoke_frame_num = 0
            self.smoke_frame_num_value.setText('0')
            self.video_info.video_path = video_path
            self.video_info.frame_rate = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            self.video_info.frame_total_num = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_info.frame_current = -1
            self.frame_current_value.setMaximum(self.video_info.frame_total_num-1)
            self.frame_rate_value.setText(str(self.video_info.frame_rate))
            self.frame_total_value.setText(str(self.video_info.frame_total_num))
            self.frame_current_value.setValue(0)
            self.video_play_pause_button.setChecked(False)
            self.video_slider.setValue(0)
            self.update_video_info()


            # Update canvas.
            self.main_timeout_respond()

            # Update toolbar.
            self.actions.open_video.setEnabled(True)

            self.actions.open_next_video.setEnabled(True)
            self.actions.open_prev_video.setEnabled(True)
            self.video_play_pause_button.setEnabled(True)
            self.video_slider.setEnabled(True)

    def load_filename(self, filename):
        ##### --Update file_list start-- #####
        video_filename_intact = filename[0]
        self.video_filename_current = video_filename_intact.split('/')[-1]
        self.video_file_dir = os.path.dirname(video_filename_intact)
        self.video_filename = []
        self.video_filelist.clear()

        files = os.listdir(self.video_file_dir)
        for f in files:

            if '.' + f.split('.')[-1] in self.video_filter:
                self.video_filename.append(f)
        self.video_filename.sort()
        self.video_filelist.addItems(self.video_filename)
        self.video_filelist.setCurrentRow(self.video_filename.index(self.video_filename_current))
        ##### --Update file_list end-- #####

        ##### --Load video and label start-- #####
        self.load_video()
        ##### --Load video and label end-- #####

    def translate_time_to_str(self, seconds):
        second = seconds % 60
        minute = (seconds // 60) % 60
        hour = seconds // 3600
        return '{}:{}:{}'.format(hour, minute, second)
    def update_video_info(self):
        self.frame_current_value.setValue(self.video_info.frame_current)
        current_time = int(self.video_info.frame_current / self.video_info.frame_rate)
        total_time = int(self.video_info.frame_total_num / self.video_info.frame_rate)
        self.video_current_time_label.setText(self.translate_time_to_str(current_time))
        self.video_total_time_label.setText(self.translate_time_to_str(total_time))
        self.video_slider.setValue(int(self.video_info.frame_current * 1000 / self.video_info.frame_total_num))

    def repaint_canvas(self):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.video_info.frame_current)
        flag, cv2_img = self.video_capture.read()
        if flag:
            qt_pixmap = cv2_img_to_qt_pixmap(cv2_img)
            if self.model is not None:
                # We can set smoke detection in the function.
                rows = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cols = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                location_list = img_to_block(rows, cols, self.model.hparams.block_size)
                self.canvas.smoke_blocks, self. canvas.motion_blocks = \
                    img_smoke_detection(self.sess, self.model, self.video_capture, self.video_info, location_list)
                if len(self.canvas.smoke_blocks) > 0:
                    self.smoke_frame_num += 1
                    self.smoke_frame_num_value.setText('{}'.format(self.smoke_frame_num))
            self.canvas.load_pixmap(qt_pixmap)
            # Reset the timer.
            self.timer_main.start(1000 / self.video_info.frame_rate)
        else:
            self.timer_main.stop()
            # self.a_open_next_video()
    ##### --Auxiliary function--end-- #####